import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from random import gauss

# path = '/home/valentina/Github/AI/FIS/dataset_1.txt'
path = '/home/valentina/Github/AI/FIS/dataset_2.txt'

#### import data ####
# with open(path) as f:
#     data = np.loadtxt(f, dtype='int', comments="#", usecols=None)

#### histogram of the dataset ####
# data = np.ravel(data)
# n, bins, patches = plt.hist(data, facecolor='purple')
# plt.show()

#### Antecedent/Consequent objects ####
arousal = ctrl.Antecedent(np.arange(-5, 5.1, 0.1), 'Arousal')
valence = ctrl.Antecedent(np.arange(-5, 5.1, 0.1), 'Valence')
angle = ctrl.Consequent(np.arange(0, 361, 1), 'Angle')
intensity = ctrl.Consequent(np.arange(0, 7.1, 0.1), 'Intensity')

#### Membership functions ####
arousal['very low'] = fuzz.gaussmf(arousal.universe, -4.7, 1.6)
arousal['low'] = fuzz.gaussmf(arousal.universe, -2, 1.5)
arousal['mid'] = fuzz.gaussmf(arousal.universe, 1, 1.6)
arousal['high'] = fuzz.gaussmf(arousal.universe, 3, 1.3)
arousal['very high'] = fuzz.gaussmf(arousal.universe, 4.9, 1.3)

valence['very low'] = fuzz.gaussmf(valence.universe, -4.8, 1.6)
valence['low'] = fuzz.gaussmf(valence.universe, -3, 1.5)
valence['mid'] = fuzz.gaussmf(valence.universe, 0, 1.3)
valence['high'] = fuzz.gaussmf(valence.universe, 3.5, 1.7)
valence['very high'] = fuzz.gaussmf(valence.universe, 4.9, 1.3)

angle['happiness'] = fuzz.gaussmf(angle.universe, 13, 53)
angle['excitement'] = fuzz.gaussmf(angle.universe, 70, 40)
angle['frustration'] = fuzz.gaussmf(angle.universe, 135, 35)
angle['depression'] = fuzz.gaussmf(angle.universe, 218, 37)
angle['boredom'] = fuzz.gaussmf(angle.universe, 280, 40)
angle['calm'] = fuzz.gaussmf(angle.universe, 350, 55)

intensity['low'] = fuzz.sigmf(intensity.universe, 2, -4)
intensity['middle'] = fuzz.pimf(intensity.universe, 2, 2.1, 4.9, 5)
intensity['high'] = fuzz.sigmf(intensity.universe, 5, 4)

arousal.view()
valence.view()
angle.view()
intensity.view()

#### arousal-valence to angle relations ####
rule1a = ctrl.Rule(valence['very low'] & arousal['very low'], angle['depression'])
rule2a = ctrl.Rule(valence['very low'] & arousal['low'], angle['depression'])
rule3a = ctrl.Rule(valence['very low'] & arousal['mid'], angle['depression'])
rule4a = ctrl.Rule(valence['very low'] & arousal['high'], angle['frustration'])
rule5a = ctrl.Rule(valence['very low'] & arousal['very high'], angle['frustration'])
rule6a = ctrl.Rule(valence['low'] & arousal['very low'], angle['boredom'])
rule7a = ctrl.Rule(valence['low'] & arousal['low'], angle['boredom'])
rule8a = ctrl.Rule(valence['low'] & arousal['low'], angle['depression'])
rule9a = ctrl.Rule(valence['low'] & arousal['mid'], angle['depression'])
rule10a = ctrl.Rule(valence['low'] & arousal['high'], angle['frustration'])
rule11a = ctrl.Rule(valence['low'] & arousal['very high'], angle['frustration'])
rule12a = ctrl.Rule(valence['mid'] & arousal['very low'], angle['boredom'])
rule13a = ctrl.Rule(valence['mid'] & arousal['low'], angle['boredom'])
rule14a = ctrl.Rule(valence['mid'] & arousal['mid'], angle['boredom'])
rule15a = ctrl.Rule(valence['mid'] & arousal['mid'], angle['depression'])
rule16a = ctrl.Rule(valence['mid'] & arousal['mid'], angle['frustration'])
rule17a = ctrl.Rule(valence['mid'] & arousal['high'], angle['excitement'])
rule18a = ctrl.Rule(valence['mid'] & arousal['high'], angle['frustration'])
rule19a = ctrl.Rule(valence['mid'] & arousal['mid'], angle['calm'])
rule20a = ctrl.Rule(valence['mid'] & arousal['mid'], angle['happiness'])
rule21a = ctrl.Rule(valence['mid'] & arousal['mid'], angle['excitement'])
rule22a = ctrl.Rule(valence['mid'] & arousal['very high'], angle['excitement'])
rule23a = ctrl.Rule(valence['high'] & arousal['very low'], angle['calm'])
rule24a = ctrl.Rule(valence['high'] & arousal['low'], angle['calm'])
rule25a = ctrl.Rule(valence['high'] & arousal['mid'], angle['calm'])
rule26a = ctrl.Rule(valence['high'] & arousal['mid'], angle['happiness'])
rule27a = ctrl.Rule(valence['high'] & arousal['mid'], angle['excitement'])
rule28a = ctrl.Rule(valence['high'] & arousal['high'], angle['happiness'])
rule29a = ctrl.Rule(valence['high'] & arousal['high'], angle['excitement'])
rule30a = ctrl.Rule(valence['high'] & arousal['very high'], angle['excitement'])
rule31a = ctrl.Rule(valence['very high'] & arousal['very low'], angle['calm'])
rule32a = ctrl.Rule(valence['very high'] & arousal['low'], angle['calm'])
rule33a = ctrl.Rule(valence['very high'] & arousal['mid'], angle['calm'])
rule34a = ctrl.Rule(valence['very high'] & arousal['mid'], angle['happiness'])
rule35a = ctrl.Rule(valence['very high'] & arousal['high'], angle['happiness'])
rule36a = ctrl.Rule(valence['very high'] & arousal['very high'], angle['excitement'])

#### arousal-valence to intensity relations ####
rule1i = ctrl.Rule(valence['very low'] & arousal['very low'], intensity['high'])
rule2i = ctrl.Rule(valence['very low'] & arousal['low'], intensity['high'])
rule3i = ctrl.Rule(valence['very low'] & arousal['mid'], intensity['middle'])
rule4i = ctrl.Rule(valence['very low'] & arousal['high'], intensity['middle'])
rule5i = ctrl.Rule(valence['very low'] & arousal['very high'], intensity['middle'])
rule6i = ctrl.Rule(valence['low'] & arousal['very low'], intensity['middle'])
rule7i = ctrl.Rule(valence['low'] & arousal['low'], intensity['middle'])
rule8i = ctrl.Rule(valence['low'] & arousal['mid'], intensity['middle'])
rule9i = ctrl.Rule(valence['low'] & arousal['high'], intensity['high'])
rule10i = ctrl.Rule(valence['low'] & arousal['very high'], intensity['high'])
rule11i = ctrl.Rule(valence['mid'] & arousal['very low'], intensity['middle'])
rule12i = ctrl.Rule(valence['mid'] & arousal['very low'], intensity['high'])
rule13i = ctrl.Rule(valence['mid'] & arousal['low'], intensity['low'])
rule14i = ctrl.Rule(valence['mid'] & arousal['low'], intensity['middle'])
rule15i = ctrl.Rule(valence['mid'] & arousal['mid'], intensity['middle'])
rule16i = ctrl.Rule(valence['mid'] & arousal['mid'], intensity['low'])
rule17i = ctrl.Rule(valence['mid'] & arousal['high'], intensity['low'])
rule18i = ctrl.Rule(valence['mid'] & arousal['high'], intensity['middle'])
rule19i = ctrl.Rule(valence['mid'] & arousal['high'], intensity['high'])
rule20i = ctrl.Rule(valence['mid'] & arousal['very high'], intensity['high'])
rule21i = ctrl.Rule(valence['high'] & arousal['very low'], intensity['low'])
rule22i = ctrl.Rule(valence['high'] & arousal['low'], intensity['high'])
rule23i = ctrl.Rule(valence['high'] & arousal['mid'], intensity['middle'])
rule24i = ctrl.Rule(valence['high'] & arousal['mid'], intensity['high'])
rule25i = ctrl.Rule(valence['high'] & arousal['high'], intensity['high'])
rule26i = ctrl.Rule(valence['high'] & arousal['very high'], intensity['high'])
rule27i = ctrl.Rule(valence['very high'] & arousal['very low'], intensity['high'])
rule28i = ctrl.Rule(valence['very high'] & arousal['low'], intensity['high'])
rule29i = ctrl.Rule(valence['very high'] & arousal['mid'], intensity['high'])
rule30i = ctrl.Rule(valence['very high'] & arousal['high'], intensity['high'])
rule31i = ctrl.Rule(valence['very high'] & arousal['very high'], intensity['high'])
# rule1a.view()

#### Control system ####
angle_ctrl = ctrl.ControlSystem([rule1a,  rule2a,  rule3a,  rule4a,  rule5a,
                                 rule6a,  rule7a,  rule8a,  rule9a,  rule10a,
                                 rule11a, rule12a, rule13a, rule14a, rule15a,
                                 rule16a, rule17a, rule18a, rule19a, rule20a,
                                 rule21a, rule22a, rule23a, rule24a, rule25a,
                                 rule26a, rule27a, rule28a, rule29a, rule30a,
                                 rule31a, rule32a, rule33a, rule34a, rule35a,
                                 rule36a])
intensity_ctrl = ctrl.ControlSystem([rule1i, rule2i,  rule3i,  rule4i,  rule5i,
                                    rule6i,  rule7i,  rule8i,  rule9i,  rule10i,
                                    rule11i, rule12i, rule13i, rule14i, rule15i,
                                    rule16i, rule17i, rule18i, rule19i, rule20i,
                                    rule21i, rule22i, rule23i, rule24i, rule25i,
                                    rule26i, rule27i, rule28i, rule29i, rule30i,
                                    rule31i])


#### simulation ####
angle_value = ctrl.ControlSystemSimulation(angle_ctrl)
intensity_value = ctrl.ControlSystemSimulation(intensity_ctrl)

def compute_AI(a, v, show=False):
    angle_value.input['Arousal'] = a
    angle_value.input['Valence'] = v
    intensity_value.input['Arousal'] = a
    intensity_value.input['Valence'] = v

    angle_value.compute()
    intensity_value.compute()

    resulting_angle = angle_value.output['Angle']
    resulting_intensity = intensity_value.output['Intensity']

    if show:
        print('The resulting angle is', resulting_angle)
        angle.view(sim=angle_value)

        print('The resulting intensity is', resulting_intensity)
        intensity.view(sim=intensity_value)

    return resulting_angle, resulting_intensity

def numbers_to_AV(num):

    emo = {
      0: [-3, 2, 'Neutral'],
      1: [3.5, -3.5, 'Anger'],
      2: [-2, -4.5, 'Disgust'],
      3: [2.3, -3.5, 'Fear'],
      4: [4, 4.5, 'Happiness'],
      5: [-2.5, -2.2, 'Sadness'],
      6: [4.6, 2.5, 'Surprise'],
      7: [4.6, -3, 'Scream'],
      8: [-3.2, -1, 'Boredom'],
      9: [-3.8, -0.5, 'Sleepy'],
      10: [0, 0, 'Unknown'],
      11: [4.9, 4.9, 'Amusement'],
      12: [2.5, -4, 'Anxiety'],
    }

    return emo.get(num, "nothing")

#### data mapping ####
def preprocess(path):
    data_a = []
    data_v = []

    with open(path) as f:
        for emo in f:
            av = numbers_to_AV(int(emo))
            a = gauss(av[0], 0.15)
            v = gauss(av[1], 0.15)

            if not -5 < a < 5:
                if abs(-5-a) < abs(5-a):
                    a = -5
                else:
                    a = 5

            if not -5 < v < 5:
                if abs(-5-v) < abs(5-v):
                    v = -5
                else:
                    v = 5

            data_a.append(a)
            data_v.append(v)

    return data_a, data_v

#### dataset in AV-space figure ####
# import pylab as lab
#
# data_a, data_v = preprocess(path)
#
# circle = plt.Circle((0, 0), 6.5, fill=False)
# ax = plt.gca()
# ax.cla()
#
# ax.set_xlim((-8, 8))
# ax.set_ylim((-8, 8))
#
# ax.plot(data_a, data_v, 'o', color='purple')
# ax.add_artist(circle)
#
# left,right = ax.get_xlim()
# low,high = ax.get_ylim()
# lab.arrow(left, 0, right -left, 0, length_includes_head = True, head_width = 0.3)
# lab.arrow(0, low, 0, high-low, length_includes_head = True, head_width = 0.3)
#
# plt.axis('off')
#
# plt.show()

#### computation of angle and intensity means ####
data_a, data_v = preprocess(path)
a_list = []
i_list = []

for a,v in zip(data_a, data_v):
    try:
        r = compute_AI(a, v)
        a_list.append(r[0])
        i_list.append(r[1])
    except:
        print('error adding:', r)

print('Mean angle:     ', str(np.mean(a_list)))
print('Mean intensity: ', str(np.mean(i_list)))
