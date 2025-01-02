import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# 定义模糊变量
current_temp = ctrl.Antecedent(np.arange(0, 41, 1), 'current_temp')
target_temp = ctrl.Antecedent(np.arange(0, 41, 1), 'target_temp')
heater_power = ctrl.Consequent(np.arange(0, 101, 1), 'heater_power')

# 定义隶属度函数
current_temp['cold'] = fuzz.trimf(current_temp.universe, [0, 0, 20])
current_temp['comfortable'] = fuzz.trimf(current_temp.universe, [15, 20, 25])
current_temp['hot'] = fuzz.trimf(current_temp.universe, [20, 40, 40])

target_temp['cold'] = fuzz.trimf(target_temp.universe, [0, 0, 20])
target_temp['comfortable'] = fuzz.trimf(target_temp.universe, [15, 20, 25])
target_temp['hot'] = fuzz.trimf(target_temp.universe, [20, 40, 40])

heater_power['low'] = fuzz.trimf(heater_power.universe, [0, 0, 50])
heater_power['medium'] = fuzz.trimf(heater_power.universe, [25, 50, 75])
heater_power['high'] = fuzz.trimf(heater_power.universe, [50, 100, 100])

# 定义模糊规则
rule1 = ctrl.Rule(current_temp['cold'] & target_temp['comfortable'], heater_power['high'])
rule2 = ctrl.Rule(current_temp['cold'] & target_temp['hot'], heater_power['high'])
rule3 = ctrl.Rule(current_temp['comfortable'] & target_temp['cold'], heater_power['medium'])
rule4 = ctrl.Rule(current_temp['comfortable'] & target_temp['comfortable'], heater_power['low'])
rule5 = ctrl.Rule(current_temp['hot'] & target_temp['cold'], heater_power['low'])

# 创建控制系统
temperature_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
temperature_sim = ctrl.ControlSystemSimulation(temperature_ctrl)

# 输入数据
temperature_sim.input['current_temp'] = 18
temperature_sim.input['target_temp'] = 22

# 计算结果
temperature_sim.compute()

print(temperature_sim.output['heater_power'])