import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from acoustics import generator
from scipy.io.wavfile import write as writewav

sampling_rate = 44100  # Hz
duration_unmod = 1  # s
duration_mod = 1
f_m = 5  # Hz

iti = [3, 5]  # s
modulation_index_db = [0, -3, -6, -9, -12] # in dB; psychometric stuff
output_name = 'AM stream_psychometricTask.pdf'
# modulation_index_db = [0, 0, 0, 0, 0] # in dB; shock training sample
# output_name = 'AM stream_shockTask.pdf'

nonAM_color = 'black'
am_color = 'black'

t_mod = np.linspace(duration_unmod, duration_unmod+duration_mod, sampling_rate * duration_mod)

f = plt.gcf()
ax = f.add_subplot(1,1,1)
last_end = 0

# Starting non-AM
cur_noise = generator.noise(sampling_rate * duration_unmod, color='white')
ax.plot(np.linspace(last_end, last_end+1, len(cur_noise)), cur_noise, color=nonAM_color, rasterized=True)
last_end += 1
final_signal = list()
for cur_idx, cur_mod in enumerate(modulation_index_db):
    # AM
    modulation_index_frac = 10 ** (cur_mod / 20)
    modulator = (1 + modulation_index_frac * np.cos(2 * np.pi * f_m * t_mod - 1.75))
    cur_noise = modulator*generator.noise(sampling_rate * duration_mod, color='white')
    ax.plot(np.linspace(last_end, last_end+1, len(cur_noise)), cur_noise, color=am_color, rasterized=True)
    last_end += 1
    final_signal.extend(cur_noise)

    if cur_idx == len(modulation_index_db)-1:  # if it's the last AM, just add one non-AM
        wait_time = 1
    else:
        wait_time = np.random.randint(iti[0], iti[1] + 1)
    for _ in np.arange(1, wait_time + 1):
        cur_noise = generator.noise(sampling_rate * duration_unmod, color='white')
        ax.plot(np.linspace(last_end, last_end+1, len(cur_noise)), cur_noise, color=nonAM_color, rasterized=True)
        last_end += 1
        final_signal.extend(cur_noise)

ax.set_xlabel('Time (s)')

f.subplots_adjust(hspace=1)
ax.set_ylim([-10, 10])
f.set_size_inches(16, 9)

f.savefig(output_name, dpi=600, transparent=True)
plt.close(f)

writewav(output_name[:-4] + '.wav', sampling_rate, np.array(final_signal))
# fig.show()


#
# plt.subplot(3,1,1)
# plt.title('Amplitude Modulation')
# plt.plot(modulator,'g')
# plt.ylabel('Amplitude')
# plt.xlabel('Message signal')
#
# plt.subplot(3,1,2)
# plt.plot(carrier, 'r')
# plt.ylabel('Amplitude')
# plt.xlabel('Carrier signal')
#
# plt.subplot(1,1,1)
# plt.plot(stream, color="purple")
# plt.ylabel('Amplitude')
# plt.xlabel('AM signal')
#
# plt.subplots_adjust(hspace=1)
# plt.rc('font', size=15)
# fig = plt.gcf()
# fig.set_size_inches(16, 9)
#
# fig.show()
# # fig.savefig('Amplitude Modulation.png', dpi=100)

#
#Carrier wave c(t)=A_c*cos(2*pi*f_c*t)
#Modulating wave m(t)=A_m*cos(2*pi*f_m*t)
#Modulated wave s(t)=A_c[1+mu*cos(2*pi*f_m*t)]cos(2*pi*f_c*t)
# AM and
# A_m = 1
# f_m = 5
# modulation_index_db = -12  # in dB

# A_c = float(input('Enter carrier amplitude: '))
# f_c = float(input('Enter carrier frquency: '))
# A_m = float(input('Enter message amplitude: '))
# f_m = float(input('Enter message frquency: '))
# modulation_index = float(input('Enter modulation index: '))
# # carrier = A_c*np.cos(2*np.pi*f_c*t)
# carrier = generator.noise(sampling_rate * duration_unmod, color='white')  # non-AM
#
# carrier_mod = generator.noise(sampling_rate * duration_mod, color='white')  # AM
#
# modulation_index_frac = 10**(modulation_index_db/20)
# modulator = (1 + modulation_index_frac * np.cos(2 * np.pi * f_m * t_mod - 1.75))
#
# product = np.append(carrier, modulator*carrier_mod)
# product = np.int16(product/np.max(np.abs(product)) * 32767)
# # sd.play(product, samplerate=sampling_rate)
# # writewav('AMtransition_' + str(modulation_index_db) + 'dB.wav', sampling_rate, product)
