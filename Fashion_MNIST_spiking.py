"""
Fashion_MNIST_spiking.py

- Backend: GPU(CuPy) / CPU(NumPy) 자동 전환
- th = 0 : Sensory adaptation
- th = 1 : VTH (threshold adaptation)
- train = 1 : 학습
- train = 0 : 테스트

VTH train 후:
  weight_th/weight.npy
  weight_th/neuron_expect.npy
  weight_th/neuron_fire_num.npy
  weight_th/theta.npy
  weight_th/Sensory_gE_max.npy

VTH test:
  위 npy 세트가 있으면 그것으로 로드, 없으면 checkpoint.npz fallback
  θ는 학습된 값 그대로 사용 (adp=0, test 중 변동 없음)
  test 결과 5개 지표를 weight_th/test_result_VTH.txt에 저장
"""

# ---------------------------------------------------------
# Backend 설정
# ---------------------------------------------------------
USE_GPU = True  # GPU 쓰기 싫으면 False

try:
    if USE_GPU:
        import cupy as cp
        xp = cp
        def asnumpy(x): return cp.asnumpy(x)
    else:
        import numpy as np
        xp = np
        def asnumpy(x): return x
except Exception:
    import numpy as np
    xp = np
    def asnumpy(x): return x
    USE_GPU = False

import numpy as np
_np = np  # CPU 전용 연산에 사용할 별칭

import matplotlib
import matplotlib.pyplot as plt
import time
import os
import pickle
from struct import unpack
from numba import njit

try:
    from scipy.stats import mode as scipy_mode
    _has_scipy = True
except Exception:
    _has_scipy = False

# ---------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------
MNIST_data_path = 'mnist/'      # Fashion-MNIST도 이 경로 사용
weight_data_path = 'random/'    # X_to_Sen.npy, Sen_in_E.npy, I_to_X.npy
weight_th_path = 'weight_th/'
weight_sen_path = 'weight_Sen/'

# ---------------------------------------------------------
# 유틸
# ---------------------------------------------------------
def most_frequent_cpu(arr1d):
    arr1d = _np.asarray(arr1d).ravel()
    if arr1d.size == 0:
        return 0
    if _has_scipy:
        res = scipy_mode(arr1d, axis=None, keepdims=True)
        try:
            return int(_np.ravel(res.mode)[0])
        except Exception:
            return int(_np.ravel(res[0])[0])
    return int(_np.bincount(arr1d).argmax())

# ---------------------------------------------------------
# 데이터 로드 (MNIST/Fashion-MNIST 공용)
# ---------------------------------------------------------
def get_labeled_data(picklename, bTrain=True):
    if os.path.isfile('%s.pickle' % picklename):
        data = pickle.load(open('%s.pickle' % picklename, 'rb'))
    else:
        if bTrain:
            images = open(MNIST_data_path + 'train-images-idx3-ubyte', 'rb')
            labels = open(MNIST_data_path + 'train-labels-idx1-ubyte', 'rb')
        else:
            images = open(MNIST_data_path + 't10k-images-idx3-ubyte', 'rb')
            labels = open(MNIST_data_path + 't10k-labels-idx1-ubyte', 'rb')

        images.read(4)
        number_of_images = unpack('>I', images.read(4))[0]
        rows = unpack('>I', images.read(4))[0]
        cols = unpack('>I', images.read(4))[0]

        labels.read(4)
        N = unpack('>I', labels.read(4))[0]

        if number_of_images != N:
            raise Exception('number of labels did not match the number of images')

        x = _np.zeros((N, rows, cols), dtype=_np.uint16)
        y = _np.zeros((N, 1), dtype=_np.uint16)

        for i in range(N):
            if i % 1000 == 0:
                print(f"load i: {i}")
            x[i] = [[unpack('>B', images.read(1))[0] for _ in range(cols)]
                    for _ in range(rows)]
            y[i] = unpack('>B', labels.read(1))[0]

        data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}
        pickle.dump(data, open('%s.pickle' % picklename, 'wb'))

    return data

# ---------------------------------------------------------
# 체크포인트 저장/로드
# ---------------------------------------------------------
def save_checkpoint(path, i,
                    weight_input_excitation,
                    neuron_expect,
                    Sensory_gE_max,
                    theta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _np.savez(
        path,
        iter=i,
        weight=_np.asarray(asnumpy(weight_input_excitation), dtype=_np.float32),
        neuron_expect=neuron_expect,
        Sensory_gE_max=_np.asarray(asnumpy(Sensory_gE_max), dtype=_np.float32),
        theta=_np.asarray(asnumpy(theta), dtype=_np.float32),
    )
    print(f"[Checkpoint 저장] iter {i + 1} -> {path}")

def load_checkpoint(path,
                    weight_input_excitation,
                    Sensory_gE_max,
                    theta):
    data = _np.load(path, allow_pickle=True)
    start_iter = int(data["iter"]) + 1

    weight_np = data["weight"].astype(_np.float32)
    neuron_expect = data["neuron_expect"]
    Sensory_gE_max_np = data["Sensory_gE_max"].astype(_np.float32)
    theta_np = data["theta"].astype(_np.float32)

    weight_input_excitation = xp.asarray(weight_np)
    Sensory_gE_max = xp.asarray(Sensory_gE_max_np)
    theta = xp.asarray(theta_np)

    print(f"[Checkpoint 로드] {path} → {start_iter} iteration부터 재시작")

    return start_iter, weight_input_excitation, neuron_expect, Sensory_gE_max, theta

# ---------------------------------------------------------
# SNN 연산부
# ---------------------------------------------------------
def poisson_spike_train(rate_cpu, interval):
    lam = (rate_cpu.astype(_np.float32) * time_step / 8.0 * interval)
    P = xp.random.uniform(0.0, 1.0,
                          (int(n_input), int(spike_rate_per_time))).astype(xp.float32)
    lam_col = xp.asarray(lam)[:, None]
    spike = xp.where(P < lam_col, 1.0, 0.0).astype(xp.float32)
    return spike

def input_spike(weight_xp, spike_xp):
    s = spike_xp.reshape(int(n_input), 1, int(spike_rate_per_time))
    return xp.matmul(weight_xp, s).astype(xp.float32)

def E_spike_gen(Excitation_potential, inhibition_potential, Sensory_gE, Sensory_gI, Inter_I_gE,
                Sensory_ge_spike, weight_excitation_inhibition, weight_inhibition_excitation,
                Sensory_gE_max, times, theta):
    Sensory_spike = xp.zeros((n_e, spike_rate_per_time), dtype=xp.uint16)
    Inter_I_spike = xp.zeros((n_e, spike_rate_per_time + 1), dtype=xp.uint16)

    for i in range(int(times)):
        v_thresh_e = -0.055 * xp.ones(n_e, dtype=xp.float32)

        # ----- Excitatory layer -----
        if i < 5:
            v_thresh_e = v_thresh_e + theta - 0.02
            spike_part_sum = xp.sum(Sensory_spike[:, 0:i], axis=1).astype(xp.uint16)
            S_not_neuron = xp.where(spike_part_sum != 0)
            S_neuron = xp.where(spike_part_sum == 0)

            I_to_E_spike_data = xp.sum(weight_inhibition_excitation * Inter_I_spike[:, i], axis=1)

            Sensory_gE = Sensory_gE * (1 - time_step / tau_syn_E) \
                         + xp.sum(Sensory_ge_spike[:, :, i], axis=0) * Sensory_gE_max
            Sensory_gI = Sensory_gI * (1 - time_step / tau_syn_I) \
                         + I_to_E_spike_data * Sensory_gI_max

            Sensory_dv = (-(Excitation_potential - v_rest_e)
                          - (Sensory_gE / gL) * (Excitation_potential - vE_E)
                          - (Sensory_gI / gL) * (Excitation_potential - vI_E)) * (time_step / tau_e)

            Excitation_potential[S_neuron] = Excitation_potential[S_neuron] + Sensory_dv[S_neuron]
            Excitation_potential[S_not_neuron] = v_rest_e

            Sensory_spike[:, i] = xp.where(v_thresh_e < Excitation_potential, 1.0, 0.0).astype(xp.float32)
            Excitation_potential = xp.where(v_thresh_e < Excitation_potential, v_reset_e,
                                            Excitation_potential).astype(xp.float32)
            Sensory_spike[S_not_neuron, i] = 0.0

            if th == 1 and adp == 1:
                theta = xp.where(Sensory_spike[:, i] == 1.0, theta + 0.00005, theta).astype(xp.float32)
                theta_dv = -theta / (10 ** 6.1)
                theta = theta + theta_dv
            elif th == 0 and adp == 1:
                Sensory_gE_max = xp.where(Sensory_spike[:, i] == 1.0,
                                          Sensory_gE_max * 0.9991, Sensory_gE_max).astype(xp.float32)
                Sensory_gE_max_dv = Sensory_gE_max / (10 ** 8.1)
                Sensory_gE_max = Sensory_gE_max + Sensory_gE_max_dv

        else:
            v_thresh_e = v_thresh_e + theta - 0.02
            I_to_E_spike_data = xp.sum(weight_inhibition_excitation * Inter_I_spike[:, i], axis=1)
            spike_part_sum = xp.sum(Sensory_spike[:, i - 5:i], axis=1)
            S_not_neuron = xp.where(spike_part_sum != 0)
            S_neuron = xp.where(spike_part_sum == 0)

            Sensory_gE = Sensory_gE * (1 - time_step / tau_syn_E) \
                         + xp.sum(Sensory_ge_spike[:, :, i], axis=0) * Sensory_gE_max
            Sensory_gI = Sensory_gI * (1 - time_step / tau_syn_I) \
                         + I_to_E_spike_data * Sensory_gI_max

            Sensory_dv = (-(Excitation_potential - v_rest_e)
                          - (Sensory_gE / gL) * (Excitation_potential - vE_E)
                          - (Sensory_gI / gL) * (Excitation_potential - vI_E)) * (time_step / tau_e)

            Excitation_potential[S_neuron] = Excitation_potential[S_neuron] + Sensory_dv[S_neuron]
            Excitation_potential[S_not_neuron] = v_rest_e

            Sensory_spike[:, i] = xp.where(v_thresh_e < Excitation_potential, 1.0, 0.0).astype(xp.float32)
            Excitation_potential = xp.where(v_thresh_e < Excitation_potential, v_reset_e,
                                            Excitation_potential).astype(xp.float32)
            Sensory_spike[S_not_neuron, i] = 0.0

            if th == 1 and adp == 1:
                theta = xp.where(Sensory_spike[:, i] == 1.0, theta + 0.00005, theta).astype(xp.float32)
                theta_dv = -theta / (10 ** 6.1)
                theta = theta + theta_dv
            elif th == 0 and adp == 1:
                Sensory_gE_max = xp.where(Sensory_spike[:, i] == 1.0,
                                          Sensory_gE_max * 0.9991, Sensory_gE_max).astype(xp.float32)
                Sensory_gE_max_dv = Sensory_gE_max / (10 ** 8.1)
                Sensory_gE_max = Sensory_gE_max + Sensory_gE_max_dv

        # ----- Inhibitory layer -----
        if i < 2:
            E_to_I_spike_data = weight_excitation_inhibition * Sensory_spike[:, i]
            spike_part_sum = xp.sum(Inter_I_spike[:, 0:i], axis=1).astype(xp.uint16)
            I_not_neuron = xp.where(spike_part_sum != 0)
            I_neuron = xp.where(spike_part_sum == 0)

            Inter_I_gE = Inter_I_gE * (1 - time_step / tau_syn_E) + E_to_I_spike_data * Inter_I_gE_max
            Inter_dv_I = (-(inhibition_potential - v_rest_i)
                          - (Inter_I_gE / gL) * (inhibition_potential - vE_I)) * (time_step / tau_e)

            inhibition_potential[I_neuron] = inhibition_potential[I_neuron] + Inter_dv_I[I_neuron]
            inhibition_potential[I_not_neuron] = v_rest_i

            Inter_I_spike[:, i + 1] = xp.where(v_thresh_i < inhibition_potential,
                                               1.0, 0.0).astype(xp.float32)
            inhibition_potential = xp.where(v_thresh_i < inhibition_potential, v_reset_i,
                                            inhibition_potential).astype(xp.float32)
            Inter_I_spike[I_not_neuron, i + 1] = 0.0
        else:
            E_to_I_spike_data = weight_excitation_inhibition * Sensory_spike[:, i]
            spike_part_sum = xp.sum(Inter_I_spike[:, i - 2:i], axis=1)
            I_not_neuron = xp.where(spike_part_sum != 0)
            I_neuron = xp.where(spike_part_sum == 0)

            Inter_I_gE = Inter_I_gE * (1 - time_step / tau_syn_E) + E_to_I_spike_data * Inter_I_gE_max
            Inter_dv_I = (-(inhibition_potential - v_rest_i)
                          - (Inter_I_gE / gL) * (inhibition_potential - vE_I)) * (time_step / tau_e)

            inhibition_potential[I_neuron] = inhibition_potential[I_neuron] + Inter_dv_I[I_neuron]
            inhibition_potential[I_not_neuron] = v_rest_i

            Inter_I_spike[:, i + 1] = xp.where(v_thresh_i < inhibition_potential,
                                               1.0, 0.0).astype(xp.float32)
            inhibition_potential = xp.where(v_thresh_i < inhibition_potential, v_reset_i,
                                            inhibition_potential).astype(xp.float32)
            Inter_I_spike[I_not_neuron, i + 1] = 0.0

    return (Excitation_potential, inhibition_potential, Sensory_gE, Sensory_gI, Inter_I_gE,
            Sensory_spike, Sensory_gE_max, theta, Inter_I_spike)

def spike_count(spike_xp):
    return xp.sum(spike_xp, axis=1)

def expect_number(count_xp, neu_expect_cpu):
    count_cpu = asnumpy(count_xp)
    maxv = _np.max(count_cpu)
    expect = neu_expect_cpu[_np.where(count_cpu == maxv)]
    expect_num = most_frequent_cpu(expect)
    return count_xp, expect_num

def winner(count_xp):
    count_cpu = asnumpy(count_xp)
    idx = _np.array(_np.where(count_cpu == _np.max(count_cpu))).reshape(-1).astype(_np.uint16)
    return idx

def normalization(weight_xp):
    weight_temp = xp.copy(weight_xp)
    Colsums = xp.sum(weight_temp, axis=0)
    Colfactors = 85.5 / Colsums
    return weight_temp * Colfactors

# ---------------------------------------------------------
# STDP (Numba)
# ---------------------------------------------------------
@njit(cache=True)
def _update_weight_single_numba(w, w_del):
    if w_del < 0.0:
        delta = 0.008 * w_del * (w ** 0.9)
        if w < abs(delta):
            return 0.0
        else:
            return w + delta
    elif w_del > 0.0:
        return w + 0.008 * w_del * ((1.0 - w) ** 0.9)
    else:
        return w

def _find_nearest_over_cpu(array, value):
    over_array = array[_np.where(array >= value)]
    if over_array.size == 0:
        return None
    return over_array[(_np.abs(over_array - value)).argmin()]

def _find_nearest_under_cpu(array, value, neuron_train_count):
    under_array = array[_np.where(array <= value)]
    if under_array.size == 0:
        return None
    under_value = under_array[(_np.abs(under_array - value)).argmin()]
    mask = (array == under_value)
    idx = _np.where(mask)
    if neuron_train_count[idx].size and neuron_train_count[idx][0] == 0:
        neuron_train_count[idx] = 1
    elif neuron_train_count[idx].size and neuron_train_count[idx][0] == 1:
        return None
    return under_value

def STDP_cpu(pre_xp, post_xp, winner_num_cpu, Weight_xp):
    pre = _np.asarray(asnumpy(pre_xp))
    post = _np.asarray(asnumpy(post_xp))
    Weight = _np.asarray(asnumpy(Weight_xp)).copy()

    a, b, c = _np.where(pre == 1)
    d, e = _np.where(post == 1)
    pre_time = _np.stack((a, b, c)).T if a.size else _np.empty((0, 3), dtype=int)
    post_time = _np.stack((d, e)).T if d.size else _np.empty((0, 2), dtype=int)
    winner_num_cpu = _np.asarray(winner_num_cpu, dtype=_np.int64)

    for i in _np.nditer(winner_num_cpu):
        pre_time_arr = pre_time[_np.where(pre_time[:, 1] == i)] if pre_time.size else _np.empty((0, 3), dtype=int)
        post_time_arr = post_time[_np.where(post_time[:, 0] == i)] if post_time.size else _np.empty((0, 2), dtype=int)
        pre_time_arr_input = _np.unique(pre_time_arr[:, 0]) if pre_time_arr.size else _np.array([], dtype=int)

        if post_time_arr.size:
            for j in _np.nditer(pre_time_arr_input):
                pre_time_arr_temp_in = pre_time_arr[_np.where(pre_time_arr[:, 0] == j)]
                setting = _np.zeros(len(pre_time_arr_temp_in))
                for k in _np.nditer(post_time_arr[:, 1]):
                    under = _find_nearest_under_cpu(pre_time_arr_temp_in[:, 2], int(k), setting)
                    if k == under:
                        w_del = 0.0
                    elif under is None:
                        over = _find_nearest_over_cpu(pre_time_arr_temp_in[:, 2], int(k))
                        if k == over:
                            w_del = 0.0
                        elif over is None:
                            w_del = -_np.exp(-1.0 / 5.0)
                        else:
                            w_del = -_np.exp((int(k) - over) / 40.0)
                    else:
                        w_del = _np.exp(-(int(k) - under) / 20.0)

                    if w_del != 0.0:
                        wij = float(Weight[int(j), int(i)])
                        Weight[int(j), int(i)] = _update_weight_single_numba(wij, float(w_del))

    return xp.asarray(Weight) if USE_GPU else Weight

# ---------------------------------------------------------
# 하이퍼파라미터 & 초기화
# ---------------------------------------------------------
n_input = 784
sim_time = 0.350
time_step = 0.001
spike_rate_per_time = 350

n_e = 625
n_i = n_e

v_rest_e = -0.065
v_rest_i = -0.06
v_reset_e = -0.080
v_reset_i = -0.075
v_thresh_i = -0.055
refrac_e = 0.005
refrac_i = 0.002

tau_e = 0.1
tau_i = 0.01
tau_syn_E = 0.001
tau_syn_I = 0.002

Sensory_gI_max = xp.ones(n_e, dtype=xp.float32)

gL = 1.0
vE_E = 0.0
vE_I = 0.0
vI_E = -0.240  # 논문 값

Excitation_potential = xp.ones(n_e, dtype=xp.float32) * v_rest_e
inhibition_potential = xp.ones(n_e, dtype=xp.float32) * v_rest_i
initial_Excitation_potential = xp.copy(Excitation_potential).astype(xp.float32)
initial_inhibition_potential = xp.copy(inhibition_potential).astype(xp.float32)

Sensory_gE = xp.zeros(n_e, dtype=xp.float32)
Sensory_gI = xp.zeros(n_e, dtype=xp.float32)
Inter_I_gE = xp.zeros(n_e, dtype=xp.float32)
Inter_I_gE_max = xp.ones(n_e, dtype=xp.float32) * 300.0

relax = xp.zeros((784, n_e, spike_rate_per_time), dtype=xp.float32)

weight_input_excitation = xp.asarray(
    _np.copy(_np.load(weight_data_path + 'X_to_Sen.npy')).astype(_np.float32)
)
weight_excitation_inhibition = xp.asarray(
    _np.copy(_np.load(weight_data_path + 'Sen_in_E.npy')).astype(_np.float32)
)
weight_inhibition_excitation = xp.asarray(
    _np.copy(_np.load(weight_data_path + 'I_to_X.npy')).astype(_np.float32)
)

training = get_labeled_data(MNIST_data_path + 'training')
testing = get_labeled_data(MNIST_data_path + 'testing', bTrain=False)

training_data = _np.array(training['x']).astype(_np.float32)
training_label = _np.array(training['y']).astype(_np.uint16)
testing_data = _np.array(testing['x']).astype(_np.float32)
testing_label = _np.array(testing['y']).astype(_np.uint16)

winner_num = 0
interval = 2.0
a = _np.zeros(10, dtype=_np.uint16)

# ---------------------------------------------------------
# 모드 설정
# ---------------------------------------------------------
train = 1   # 1: 학습, 0: 테스트
th = 1      # 0: Sensory, 1: VTH

training_iter_all = 180000
training_iter = 60000 if th == 1 else training_iter_all

start_time_all = time.time()

# =========================================================
# Train 모드
# =========================================================
if train == 1:
    total_count = 0.0
    performance_count = 0.0
    adp = 1  # 학습 시 적응 on

    theta = xp.ones(n_e, dtype=xp.float32) * 0.02
    Sensory_gE_max = xp.ones(n_e, dtype=xp.float32)

    neuron_expect = _np.zeros(n_e, dtype=_np.uint16)
    neuron_fire_num = _np.zeros((n_e, 10), dtype=_np.uint16)

    window_spike_sum = 0.0
    window_noi_sum = 0.0
    window_wta_size_sum = 0.0
    window_single_wta = 0
    window_sample_count = 0

    checkpoint_dir = 'weight_th' if th == 1 else 'weight_Sen'
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.npz')

    if os.path.exists(checkpoint_path):
        start_iter, weight_input_excitation, neuron_expect, Sensory_gE_max, theta = \
            load_checkpoint(checkpoint_path, weight_input_excitation, Sensory_gE_max, theta)
    else:
        start_iter = 0
        print('새로 처음부터 학습 시작')

    print('start train (VTH)' if th == 1 else 'start train (Sensory)')

    for i in range(start_iter, training_iter):
        real_num = int(training_label[i % 60000][0])
        data_cpu = training_data[i % 60000, :, :].reshape(n_input)

        count = xp.zeros(n_e, dtype=xp.float32)
        start_Excitation_potential = xp.copy(Excitation_potential).astype(xp.float32)
        start_inhibition_potential = xp.copy(inhibition_potential).astype(xp.float32)
        start_theta = xp.copy(theta).astype(xp.float32)

        weight_input_excitation = normalization(weight_input_excitation)

        # weight 시각화
        if i == 0 or i % 10000 == 9999:
            fig, axes = plt.subplots(nrows=25, ncols=25, figsize=(25, 25))
            weight_show = asnumpy(xp.reshape(weight_input_excitation, (784, n_e)))
            for j, ax in zip(range(n_e), axes.flat):
                ws = weight_show[:, j].reshape(28, 28)
                im = ax.imshow(ws, vmin=0, vmax=1,
                               cmap=matplotlib.colormaps.get_cmap('hot_r'))
                ax.set_yticks([])
                ax.set_xticks([])
            fig.subplots_adjust(right=0.835)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(im, cax=cbar_ax)
            save_dir = weight_th_path if th == 1 else weight_sen_path
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"{i + 1}.jpg"),
                        dpi=300, bbox_inches='tight')
            plt.close(fig)

        # 최소 5 spikes까지 이미지 반복
        while float(xp.sum(count)) < 5.0:
            if i == 0:
                Excitation_potential = xp.copy(initial_Excitation_potential)
                inhibition_potential = xp.copy(initial_inhibition_potential)
                theta = xp.copy(start_theta)
            else:
                Excitation_potential = xp.copy(start_Excitation_potential)
                inhibition_potential = xp.copy(start_inhibition_potential)
                theta = xp.copy(start_theta)

            current_spike = poisson_spike_train(data_cpu, interval)
            Sensory_ge_spike = input_spike(weight_input_excitation, current_spike)

            (Excitation_potential, inhibition_potential, Sensory_gE, Sensory_gI, Inter_I_gE,
             Sensory_spike, Sensory_gE_max, theta, Inter_I_spikes) = E_spike_gen(
                Excitation_potential, inhibition_potential, Sensory_gE, Sensory_gI, Inter_I_gE,
                Sensory_ge_spike, weight_excitation_inhibition, weight_inhibition_excitation,
                Sensory_gE_max, 300, theta)

            count = spike_count(Sensory_spike)
            if float(xp.sum(count)) < 5.0:
                interval += 1.0

        count, num_expect = expect_number(count, neuron_expect)
        winner_num = winner(count)

        # 통계
        sample_exc_spikes = float(xp.sum(count))
        sample_inh_spikes = float(xp.sum(Inter_I_spikes))
        sample_spikes = sample_exc_spikes + sample_inh_spikes
        sample_noi = int(xp.count_nonzero(count))
        sample_wta_size = int(winner_num.size)
        sample_single_flag = 1 if sample_wta_size == 1 else 0

        window_spike_sum += sample_spikes
        window_noi_sum += sample_noi
        window_wta_size_sum += sample_wta_size
        window_single_wta += sample_single_flag
        window_sample_count += 1

        neuron_fire_num[winner_num, real_num] += 1
        a[real_num] += 1
        if float(xp.sum(count)) != 0.0:
            interval = 2.0

        pre_spike = xp.where(Sensory_ge_spike != 0.0, 1.0, 0.0)
        weight_input_excitation = STDP_cpu(pre_spike, Sensory_spike,
                                           winner_num, weight_input_excitation)

        (_Excitation_potential, _inhibition_potential, relax_Sensory_gE, relax_Sensory_gI, Inter_I_gE,
         relax_Sensory_spike, relax_Sensory_gE_max, relax_theta, relax_Inter_I_spike) = E_spike_gen(
            Excitation_potential, inhibition_potential, Sensory_gE, Sensory_gI, Inter_I_gE,
            relax, weight_excitation_inhibition, weight_inhibition_excitation,
            Sensory_gE_max, 50, theta)

        total_count += float(xp.sum(count)) + float(xp.sum(Inter_I_spikes))
        if num_expect == real_num:
            performance_count += 1.0

        # ---- 10000 iter마다 로그 + checkpoint ----
        if i % 10000 == 9999:
            neuron_expect = _np.argmax((neuron_fire_num / _np.maximum(a, 1)), axis=1)
            acc = (performance_count / 10000.0) * 100.0

            avg_spikes = window_spike_sum / max(window_sample_count, 1)
            avg_noi = window_noi_sum / max(window_sample_count, 1)
            avg_wta_size = window_wta_size_sum / max(window_sample_count, 1)
            single_wta_ratio = (window_single_wta / max(window_sample_count, 1)) * 100.0

            msg1 = f"[iter {i + 1}] accuracy = {acc:.2f} %"
            msg_stats = (
                f"  avg spikes per image (exc+inh): {avg_spikes:.2f}\n"
                f"  avg NOI: {avg_noi:.2f}\n"
                f"  avg WTA winner size: {avg_wta_size:.2f}\n"
                f"  single-WTA ratio: {single_wta_ratio:.2f} %"
            )
            if USE_GPU and xp is cp:
                xp.cuda.Stream.null.synchronize()
            elapsed = time.time() - start_time_all
            msg2 = f"train time = {elapsed:.1f} s"

            print(msg1)
            print(msg_stats)
            print(msg2)

            performance_count = 0.0
            neuron_fire_num = _np.zeros((n_e, 10), dtype=_np.uint16)
            a = _np.zeros(10, dtype=_np.uint16)

            log_dir = checkpoint_dir
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, 'train_log.txt')
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(msg1 + '\n')
                f.write(msg_stats + '\n')
                f.write(msg2 + '\n')

            save_checkpoint(
                checkpoint_path,
                i,
                weight_input_excitation,
                neuron_expect,
                Sensory_gE_max,
                theta
            )
            checkpoint_path_iter = os.path.join(
                checkpoint_dir,
                f'checkpoint_{i + 1:08d}.npz'
            )
            save_checkpoint(
                checkpoint_path_iter,
                i,
                weight_input_excitation,
                neuron_expect,
                Sensory_gE_max,
                theta
            )

            window_spike_sum = 0.0
            window_noi_sum = 0.0
            window_wta_size_sum = 0.0
            window_single_wta = 0
            window_sample_count = 0

    # ---- 학습 종료 후 npy 저장 ----
    if th == 1:
        os.makedirs(weight_th_path, exist_ok=True)
        _np.save(os.path.join(weight_th_path, 'weight.npy'), asnumpy(weight_input_excitation))
        _np.save(os.path.join(weight_th_path, 'neuron_expect.npy'), neuron_expect)
        _np.save(os.path.join(weight_th_path, 'neuron_fire_num.npy'), neuron_fire_num)
        _np.save(os.path.join(weight_th_path, 'theta.npy'), asnumpy(theta))
        _np.save(os.path.join(weight_th_path, 'Sensory_gE_max.npy'), asnumpy(Sensory_gE_max))
    else:
        os.makedirs(weight_sen_path, exist_ok=True)
        _np.save(os.path.join(weight_sen_path, 'weight.npy'), asnumpy(weight_input_excitation))
        _np.save(os.path.join(weight_sen_path, 'neuron_expect.npy'), neuron_expect)
        _np.save(os.path.join(weight_sen_path, 'neuron_fire_num.npy'), neuron_fire_num)
        _np.save(os.path.join(weight_sen_path, 'Sensory_gE_max.npy'), asnumpy(Sensory_gE_max))

    print('train done, time =', time.time() - start_time_all)

# =========================================================
# Test 모드
# =========================================================
else:
    adp = 0  # 테스트에서 적응 OFF
    performance_count = 0
    interval = 2.0

    test_spike_sum = 0.0
    test_noi_sum = 0.0
    test_wta_size_sum = 0.0
    test_single_wta = 0
    test_sample_count = 0

    checkpoint_dir = 'weight_th' if th == 1 else 'weight_Sen'

    if th == 1:
        # VTH: npy 우선 사용, 없으면 checkpoint fallback
        w_file = os.path.join(checkpoint_dir, 'weight.npy')
        exp_file = os.path.join(checkpoint_dir, 'neuron_expect.npy')
        gmax_file = os.path.join(checkpoint_dir, 'Sensory_gE_max.npy')
        theta_file = os.path.join(checkpoint_dir, 'theta.npy')

        if (os.path.exists(w_file) and os.path.exists(exp_file)
                and os.path.exists(gmax_file) and os.path.exists(theta_file)):
            print("VTH test: npy 로드 (weight/theta/gE_max)")
            weight_input_excitation = xp.asarray(_np.load(w_file).astype(_np.float32))
            neuron_expect = _np.load(exp_file)
            Sensory_gE_max = xp.asarray(_np.load(gmax_file).astype(_np.float32))
            theta = xp.asarray(_np.load(theta_file).astype(_np.float32))
        else:
            print("VTH test: npy 없음 → checkpoint.npz에서 로드")
            ckpt_path = os.path.join(checkpoint_dir, 'checkpoint.npz')
            ckpt = _np.load(ckpt_path, allow_pickle=True)
            weight_input_excitation = xp.asarray(ckpt['weight'].astype(_np.float32))
            neuron_expect = ckpt['neuron_expect']
            Sensory_gE_max = xp.asarray(ckpt['Sensory_gE_max'].astype(_np.float32))
            theta = xp.asarray(ckpt['theta'].astype(_np.float32))

        print("start test - VTH mode (θ from train, fixed during test)")
    else:
        # Sensory: npy 우선, 없으면 checkpoint fallback (θ는 고정)
        w_file = os.path.join(checkpoint_dir, 'weight.npy')
        exp_file = os.path.join(checkpoint_dir, 'neuron_expect.npy')
        gmax_file = os.path.join(checkpoint_dir, 'Sensory_gE_max.npy')

        if os.path.exists(w_file) and os.path.exists(exp_file) and os.path.exists(gmax_file):
            print("Sensory test: npy 로드 (weight/gE_max)")
            weight_input_excitation = xp.asarray(_np.load(w_file).astype(_np.float32))
            neuron_expect = _np.load(exp_file)
            Sensory_gE_max = xp.asarray(_np.load(gmax_file).astype(_np.float32))
            theta = xp.ones(n_e, dtype=xp.float32) * 0.02
        else:
            print("Sensory test: npy 없음 → checkpoint.npz에서 로드")
            ckpt_path = os.path.join(checkpoint_dir, 'checkpoint.npz')
            ckpt = _np.load(ckpt_path, allow_pickle=True)
            weight_input_excitation = xp.asarray(ckpt['weight'].astype(_np.float32))
            neuron_expect = ckpt['neuron_expect']
            Sensory_gE_max = xp.asarray(ckpt['Sensory_gE_max'].astype(_np.float32))
            theta = xp.asarray(ckpt['theta'].astype(_np.float32))

        print("start test - Sensory mode")

    start_test_time = time.time()

    for i in range(10000):
        start_Excitation_potential = xp.copy(Excitation_potential)
        start_inhibition_potential = xp.copy(inhibition_potential)
        start_theta = xp.copy(theta)

        count = xp.zeros(n_e, dtype=xp.float32)

        data_cpu = testing_data[i % 10000, :, :].reshape(n_input).astype(_np.float32)
        real_num = int(testing_label[i % 10000, 0])

        while float(xp.sum(count)) < 5.0:
            if i == 0:
                Excitation_potential = xp.copy(initial_Excitation_potential)
                inhibition_potential = xp.copy(initial_inhibition_potential)
                theta = xp.copy(start_theta)
            else:
                Excitation_potential = xp.copy(start_Excitation_potential)
                inhibition_potential = xp.copy(start_inhibition_potential)
                theta = xp.copy(start_theta)

            current_spike = poisson_spike_train(data_cpu, interval)
            Sensory_ge_spike = input_spike(weight_input_excitation, current_spike)

            (Excitation_potential, inhibition_potential,
             Sensory_gE, Sensory_gI, Inter_I_gE,
             Sensory_spike, Sensory_gE_max, theta,
             Inter_I_spikes) = E_spike_gen(
                Excitation_potential, inhibition_potential,
                Sensory_gE, Sensory_gI, Inter_I_gE,
                Sensory_ge_spike,
                weight_excitation_inhibition, weight_inhibition_excitation,
                Sensory_gE_max,
                350, theta)

            count = spike_count(Sensory_spike)
            if float(xp.sum(count)) < 5.0:
                interval += 1.0

        count, num_expect = expect_number(count, neuron_expect)
        winner_num = winner(count)

        sample_exc_spikes = float(xp.sum(count))
        sample_inh_spikes = float(xp.sum(Inter_I_spikes))
        sample_spikes = sample_exc_spikes + sample_inh_spikes
        sample_noi = int(xp.count_nonzero(count))
        sample_wta_size = int(winner_num.size)
        sample_single_flag = 1 if sample_wta_size == 1 else 0

        test_spike_sum += sample_spikes
        test_noi_sum += sample_noi
        test_wta_size_sum += sample_wta_size
        test_single_wta += sample_single_flag
        test_sample_count += 1

        # relax용 300 step (원본 구조 유지)
        (Excitation_potential, inhibition_potential,
         Sensory_gE, Sensory_gI, Inter_I_gE,
         Sensory_spike, Sensory_gE_max, theta,
         Inter_I_spikes) = E_spike_gen(
            Excitation_potential, inhibition_potential,
            Sensory_gE, Sensory_gI, Inter_I_gE,
            Sensory_ge_spike,
            weight_excitation_inhibition, weight_inhibition_excitation,
            Sensory_gE_max,
            300, theta)

        if float(xp.sum(count)) > 5.0:
            interval = 2.0

        if num_expect == real_num:
            performance_count += 1

        if (i + 1) % 1000 == 0:
            if USE_GPU and xp is cp:
                xp.cuda.Stream.null.synchronize()
            elapsed = time.time() - start_test_time
            acc_so_far = performance_count / float(i + 1) * 100.0
            print(f"[TEST {i + 1}/10000] accuracy = {acc_so_far:.2f} %, time = {elapsed/60:.1f} min")

    # 최종 통계 및 저장
    if USE_GPU and xp is cp:
        xp.cuda.Stream.null.synchronize()
    total_elapsed = time.time() - start_test_time
    final_acc = performance_count / 10000.0 * 100.0

    avg_spikes = test_spike_sum / max(test_sample_count, 1)
    avg_noi = test_noi_sum / max(test_sample_count, 1)
    avg_wta_size = test_wta_size_sum / max(test_sample_count, 1)
    single_wta_ratio = 100.0 * test_single_wta / max(test_sample_count, 1)

    print('FINAL accuracy = ', final_acc, '%')
    print('avg spikes per image (exc+inh):', avg_spikes)
    print('avg NOI:', avg_noi)
    print('avg WTA winner size:', avg_wta_size)
    print('single-WTA ratio:', single_wta_ratio, '%')
    print('total test time = ', total_elapsed, 's (', total_elapsed / 60.0, 'min )')

    if th == 1:
        out_dir = weight_th_path
        out_name = 'test_result_VTH.txt'
    else:
        out_dir = weight_sen_path
        out_name = 'test_result_Sensory.txt'

    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, out_name)
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write(f"accuracy: {final_acc:.4f} %\n")
        f.write(f"avg spikes per image (exc+inh): {avg_spikes:.4f}\n")
        f.write(f"avg NOI: {avg_noi:.4f}\n")
        f.write(f"avg WTA winner size: {avg_wta_size:.4f}\n")
        f.write(f"single-WTA ratio: {single_wta_ratio:.4f} %\n")

    print("saved test summary to:", result_path)
