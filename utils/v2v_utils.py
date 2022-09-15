
import os
import numpy.random as s
from functools import cmp_to_key

class Car_v2v_1:
    def __init__(self, args, car_id, cal_power, t_in, t_out,t, road_len=400):
        self.args = args
        self.car_id = car_id
        self.cal_power = cal_power
        self.t_in = t_in
        self.t_out = t_out
        self.speed = road_len / (t_out - t_in)
        self.road_len = road_len
        self.t = t
        self.position = (self.t - self.t_in) * self.speed
        if self.position < 0 or self.position > road_len:
            print("BUG: CAR OUT OF ROAD")
        self.train_list = []
        self.car_state = "download"
        self.car_progress  = 0
    
    def reset(self):
        self.train_list = []
        self.car_state = "download"
        self.car_progress  = 0
        
    def __repr__(self):
        return str((self.car_id, self.position))    
    
    def update_state(self, dt=0.05):
        self.t += dt
        self.position += self.speed*dt
        
        if self.car_state == "download":
            self.car_progress += dt/self.args.delay_download
            if self.car_progress >= 1:
                self.car_state = "train"
                self.train_list.append(["model_"+str(self.car_id),{self.car_id:0}]) # ["model_1",{"car_1":13.3, "car_2":3.2}]
        elif self.car_state == "train":
            if self.car_id in self.train_list[0][1].keys():
                self.train_list[0][1][self.car_id] += self.cal_power*dt
            else:
                self.train_list[0][1][self.car_id] = self.cal_power*dt
            if sum([int(car_iter) for car_iter in self.train_list[0][1].values()]) >= self.args.local_iter:
                self.car_state = "upload"
        elif self.car_state == "upload":
            self.car_progress += dt/self.args.delay_upload
            if self.car_progress >= 1:
                self.car_state = "idle"
                return self.train_list.pop(0) # ["model_1",{"car_1":13.3, "car_2":3.2}]
        elif self.car_state == "idle":
            if len(self.train_list) > 0:
                self.car_state = "train"
        else:
            exit("BUG")
        return None
    
    def send_model_message(self):
        if len(self.train_list) > 0:
            model_message = self.train_list[0]
            return model_message
        else:
            return None
    
    def receive_model_message(self, model_message):
        self.train_list.append(model_message)
        self.sort_train_list()
        
    def sort_train_list(self):
        def cmp_func(a,b):
            a1 = sum([int(i) for i in a[1].values()])
            b1 = sum([int(i) for i in b[1].values()])
            if a1 < b1:
                return 1
            elif a1 > b1:
                return -1
            else:
                return 0
        self.train_list.sort(key=cmp_to_key(cmp_func), reverse=False)
    
    
class Link_v2v_1:
    def __init__(self, args, transmitter, receiver, model_message):
        self.args = args
        self.transmitter = transmitter
        self.receiver = receiver
        self.message = model_message
        self.link_progress = 0
        self.link_state = "active"
    
    def update_state(self, dt=0.05):
        self.link_progress += dt/self.args.delay_upload
        if self.link_progress >= 1:
            self.receiver.receive_model_message(self.message)
            self.link_state = "over"
            #return self.message, self.receiver.car_id
    
class Road_v2v_1:
    def __init__(self, args, road_len, car_tripinfo, V2V_range=400, dt=0.05):
        self.args = args
        self.road_len = road_len
        self.dt = dt
        self.V2V_range = V2V_range
        self.t = 0 # 当前时间
        self.round = 0 # 当前轮次
        self.cars_passed = 0 # 当前已到达过道路的车辆数
        tripInfo_dict = {}
        for i in car_tripinfo:
            tripInfo_dict[i['id']] = [float(i['depart']),float(i['arrival'])]
            
        T = args.num_steps                      # total_training_time
        T_round = args.round_duration           # duration of a round

        num_rounds = int(T/T_round)
        car_calculation_power = {} # 每辆车的计算能力
        cars_dict_per_round = {} # 每轮中曾在道路上行驶的车辆集合
        
        MU_local_train = args.local_iter * args.mu_local_train   # param of shift exponential distribution function for local training delay
        BETA_local_train = args.local_iter * args.beta_local_train   # param of shift exponential distribution function for local training delay
        for k in tripInfo_dict.keys():
            car_calculation_power[k] = args.local_iter / (float(s.exponential(BETA_local_train,(1,)))+MU_local_train)
            
        for r in range(num_rounds):
            cars_dict_per_round[r] = []
        for k,v in tripInfo_dict.items():
            r1 = int(v[0]/T_round)
            r2 = int(v[1]/T_round)
            for r in range(r1,r2+1):
                if r in cars_dict_per_round.keys():
                    cars_dict_per_round[r].append((k,v))
                    
        self.car_calculation_power = car_calculation_power
        self.cars_dict_per_round = cars_dict_per_round
        self.tripInfo_dict = tripInfo_dict
        self.cars_on_road = []
        self.FL_table = {self.round:{}}
        self.car_last_out = None
        self.link = None
        
    def run_a_step(self):
        self.t += self.dt
        if self.link != None and self.link.link_state == "active":
            self.link.update_state(dt=self.dt)
        
        for idx,car in enumerate(self.cars_on_road):
            model_message = car.update_state(dt=self.dt) # ["model_1",{"car_1":13.3, "car_2":3.2}]
            if model_message != None:
                model_name, training_message = model_message
                self.FL_table[self.round][model_name] = training_message
            if car.position > self.road_len:
                self.car_last_out = self.cars_on_road.pop(idx)
                if len(self.cars_on_road) > 0 and self.cars_on_road[0].position > self.road_len-self.V2V_range and len(self.car_last_out.train_list) > 0:
                    self.link = Link_v2v_1(args=self.args, transmitter=self.car_last_out, 
                                     receiver=self.cars_on_road[0], 
                                     model_message = self.car_last_out.send_model_message())
                else:
                    self.link = None
        
        # 加入新到达的车辆
        new_car_id = "car_"+str(self.cars_passed)
        if new_car_id in self.tripInfo_dict.keys() and  self.tripInfo_dict[new_car_id][0] <= self.t:
            cal_power = self.car_calculation_power[new_car_id]
            t_in = self.tripInfo_dict[new_car_id][0]
            t_out = self.tripInfo_dict[new_car_id][1]
            self.cars_on_road.append(Car_v2v_1(args=self.args, car_id=new_car_id,cal_power=cal_power, t_in=t_in, t_out=t_out, t=self.t, road_len=self.road_len))
            self.cars_passed += 1
            
        # 判断是否到新的一轮
        if self.t > (self.round + 1) * self.args.round_duration:
            self.round += 1
            self.FL_table[self.round]={}
            for idx,car in enumerate(self.cars_on_road):
                car.reset()
                self.link = None


class Car_v2v_2:
    def __init__(self, args, car_id, cal_power, t_in, t_out,t, road_len=400):
        self.args = args
        self.car_id = car_id
        self.cal_power = cal_power
        self.t_in = t_in
        self.t_out = t_out
        self.speed = road_len / (t_out - t_in)
        self.road_len = road_len
        self.t = t
        self.position = (self.t - self.t_in) * self.speed
        if self.position < 0 or self.position > road_len:
            print("BUG: CAR OUT OF ROAD")
        self.train_list = []
        self.car_state = "download"
        self.car_progress  = 0
    
    def reset(self):
        self.train_list = []
        self.car_state = "download"
        self.car_progress  = 0
        
    def __repr__(self):
        return str((self.car_id, self.position))
        
    
    def update_state(self, dt=0.05):
        self.t += dt
        self.position += self.speed*dt
        
        if self.car_state == "download":
            self.car_progress += dt/self.args.delay_download
            if self.car_progress >= 1:
                self.car_state = "train"
                self.train_list.append(["model_"+str(self.car_id),{self.car_id:0}]) # ["model_1",{"car_1":13.3, "car_2":3.2}]
        elif self.car_state == "train":
            if self.car_id in self.train_list[0][1].keys():
                self.train_list[0][1][self.car_id] += self.cal_power*dt
            else:
                self.car_state = "upload"
            if sum([int(car_iter) for car_iter in self.train_list[0][1].values()]) >= self.args.local_iter:
                self.car_state = "upload"
        elif self.car_state == "upload":
            self.car_progress += dt/self.args.delay_upload
            if self.car_progress >= 1:
                self.car_state = "idle"
                return self.train_list.pop(0) # ["model_1",{"car_1":13.3, "car_2":3.2}]
        elif self.car_state == "idle":
            if len(self.train_list) > 0:
                self.car_state = "train"
        else:
            exit("BUG")
        return None
    
    def send_model_message(self):
        if len(self.train_list) > 0:
            model_message = self.train_list[0]
            return model_message
        else:
            return None
    
    def receive_model_message(self, model_message):
        self.train_list.append(model_message)
        self.sort_train_list()
        
    def sort_train_list(self):
        def cmp_func(a,b):
            a1 = sum([int(i) for i in a[1].values()])
            b1 = sum([int(i) for i in b[1].values()])
            if a1 < b1:
                return 1
            elif a1 > b1:
                return -1
            else:
                return 0
        self.train_list.sort(key=cmp_to_key(cmp_func), reverse=False)
    
    
class Link_v2v_2:
    def __init__(self, args, transmitter, receiver, model_message):
        self.args = args
        self.transmitter = transmitter
        self.receiver = receiver
        self.message = model_message
        self.link_progress = 0
        self.link_state = "active"
    
    def update_state(self, dt=0.05):
        self.link_progress += dt/self.args.delay_upload
        if self.link_progress >= 1:
            self.receiver.receive_model_message(self.message)
            self.link_state = "over"
            #return self.message, self.receiver.car_id
    
class Road_v2v_2:
    def __init__(self, args, road_len, car_tripinfo, V2V_range=400, dt=0.05):
        self.args = args
        self.road_len = road_len
        self.dt = dt
        self.V2V_range = V2V_range
        self.t = 0 # 当前时间
        self.round = 0 # 当前轮次
        self.cars_passed = 0 # 当前已到达过道路的车辆数
        tripInfo_dict = {}
        for i in car_tripinfo:
            tripInfo_dict[i['id']] = [float(i['depart']),float(i['arrival'])]
            
        T = args.num_steps                      # total_training_time
        T_round = args.round_duration           # duration of a round
        mu_local_train = args.mu_local_train   # param of shift exponential distribution function for local training delay
        beta_local_train = args.beta_local_train   # param of shift exponential distribution function for local training delay


        num_rounds = int(T/T_round)
        car_calculation_power = {} # 每辆车的计算能力
        cars_dict_per_round = {} # 每轮中曾在道路上行驶的车辆集合
        
        MU_local_train = args.local_iter * args.mu_local_train   # param of shift exponential distribution function for local training delay
        BETA_local_train = args.local_iter * args.beta_local_train   # param of shift exponential distribution function for local training delay
        for k in tripInfo_dict.keys():
            car_calculation_power[k] = args.local_iter / (float(s.exponential(BETA_local_train,(1,)))+MU_local_train)
            
        for r in range(num_rounds):
            cars_dict_per_round[r] = []
        for k,v in tripInfo_dict.items():
            r1 = int(v[0]/T_round)
            r2 = int(v[1]/T_round)
            for r in range(r1,r2+1):
                if r in cars_dict_per_round.keys():
                    cars_dict_per_round[r].append((k,v))
                    
        self.car_calculation_power = car_calculation_power
        self.cars_dict_per_round = cars_dict_per_round
        self.tripInfo_dict = tripInfo_dict
        self.cars_on_road = []
        self.FL_table = {self.round:{}}
        self.car_last_out = None
        self.link = None
        
    def run_a_step(self):
        self.t += self.dt
        if self.link != None and self.link.link_state == "active":
            self.link.update_state(dt=self.dt)
        
        for idx,car in enumerate(self.cars_on_road):
            model_message = car.update_state(dt=self.dt) # ["model_1",{"car_1":13.3, "car_2":3.2}]
            if model_message != None:
                model_name, training_message = model_message
                self.FL_table[self.round][model_name] = training_message
            if car.position > self.road_len:
                self.car_last_out = self.cars_on_road.pop(idx)
                if len(self.cars_on_road) > 0 and self.cars_on_road[0].position > self.road_len-self.V2V_range and len(self.car_last_out.train_list) > 0:
                    self.link = Link_v2v_2(args=self.args, transmitter=self.car_last_out, 
                                     receiver=self.cars_on_road[0], 
                                     model_message = self.car_last_out.send_model_message())
                else:
                    self.link = None
        
                    
        # 加入新到达的车辆
        new_car_id = "car_"+str(self.cars_passed)
        if new_car_id in self.tripInfo_dict.keys() and  self.tripInfo_dict[new_car_id][0] <= self.t:
            cal_power = self.car_calculation_power[new_car_id]
            t_in = self.tripInfo_dict[new_car_id][0]
            t_out = self.tripInfo_dict[new_car_id][1]
            self.cars_on_road.append(Car_v2v_2(args=self.args, car_id=new_car_id,cal_power=cal_power, t_in=t_in, t_out=t_out, t=self.t, road_len=self.road_len))
            self.cars_passed += 1
            
        # 判断是否到新的一轮
        if self.t > (self.round + 1) * self.args.round_duration:
            self.round += 1
            self.FL_table[self.round]={}
            for idx,car in enumerate(self.cars_on_road):
                car.reset()
                self.link = None


class Car_v2v_3:
    def __init__(self, args, car_id, cal_power, t_in, t_out,t, road_len=400):
        self.args = args
        self.car_id = car_id
        self.cal_power = cal_power
        self.t_in = t_in
        self.t_out = t_out
        self.speed = road_len / (t_out - t_in)
        self.road_len = road_len
        self.t = t
        self.position = (self.t - self.t_in) * self.speed
        if self.position < 0 or self.position > road_len:
            print("BUG: CAR OUT OF ROAD")
        self.train_list = []
        self.car_state = "download"
        self.car_progress  = 0
    
    def reset(self):
        self.train_list = []
        self.car_state = "download"
        self.car_progress  = 0
        
    def __repr__(self):
        return str((self.car_id, self.position))
        
    
    def update_state(self, dt=0.05):
        self.t += dt
        self.position += self.speed*dt
        
        if self.car_state == "download":
            self.car_progress += dt/self.args.delay_download
            if self.car_progress >= 1:
                self.car_state = "train"
                self.train_list.append(["model_"+str(self.car_id),{self.car_id:0}]) # ["model_1",{"car_1":13.3, "car_2":3.2}]
        elif self.car_state == "train":
            if self.car_id in self.train_list[0][1].keys():
                self.train_list[0][1][self.car_id] += self.cal_power*dt
            else:
                self.car_state = "upload"
            if sum([int(car_iter) for car_iter in self.train_list[0][1].values()]) >= self.args.local_iter:
                self.car_state = "upload"
        elif self.car_state == "upload":
            self.car_progress += dt/self.args.delay_upload
            if self.car_progress >= 1:
                self.car_state = "idle"
                return self.train_list.pop(0) # ["model_1",{"car_1":13.3, "car_2":3.2}]
        elif self.car_state == "idle":
            if len(self.train_list) > 0:
                self.car_state = "train"
        else:
            exit("BUG")
        return None

    def car_last_update_state(self, dt=0.05):
        self.t += dt
        self.position += self.speed*dt
        
        if len(self.train_list) == 0:
            self.car_state = "idle"
        elif self.car_state == "download":
            self.car_state = "call_v2v"
        elif self.car_state == "train":
            if self.car_id in self.train_list[0][1].keys():
                self.train_list[0][1][self.car_id] += self.cal_power*dt
            else:
                self.car_state = "call_v2v"
            if sum([int(car_iter) for car_iter in self.train_list[0][1].values()]) >= self.args.local_iter:
                self.car_state = "call_v2v"
        elif self.car_state == "upload":
            self.car_state = "call_v2v"
        elif self.car_state == "idle":
            self.car_state = "train"
        elif self.car_state == "call_v2v":
            self.car_state = "call_v2v"
        elif self.car_state == "v2v_ing":
            self.car_state = "v2v_ing"
        else:
            exit("BUG")
        return None

    def send_model_message(self):
        if len(self.train_list) > 0:
            model_message = self.train_list.pop(0)
            return model_message
        else:
            return None
    
    def receive_model_message(self, model_message):
        self.train_list.append(model_message)
        self.sort_train_list()
        
    def sort_train_list(self):
        def cmp_func(a,b):
            a1 = sum([int(i) for i in a[1].values()])
            b1 = sum([int(i) for i in b[1].values()])
            if a1 < b1:
                return 1
            elif a1 > b1:
                return -1
            else:
                return 0
        self.train_list.sort(key=cmp_to_key(cmp_func), reverse=False)
    
    
class Link_v2v_3:
    def __init__(self, args, transmitter, receiver, model_message):
        self.args = args
        self.transmitter = transmitter
        self.receiver = receiver
        self.message = model_message
        self.link_progress = 0
        self.link_state = "active"
    
    def update_state(self, dt=0.05):
        self.link_progress += dt/self.args.delay_upload
        if self.link_progress >= 1:
            self.receiver.receive_model_message(self.message)
            self.transmitter.car_state = "idle"
            self.link_state = "over"
            #return self.message, self.receiver.car_id
    
class Road_v2v_3:
    def __init__(self, args, road_len, car_tripinfo, V2V_range=400, dt=0.05):
        self.args = args
        self.road_len = road_len
        self.dt = dt
        self.V2V_range = V2V_range
        self.t = 0 # 当前时间
        self.round = 0 # 当前轮次
        self.cars_passed = 0 # 当前已到达过道路的车辆数
        tripInfo_dict = {}
        for i in car_tripinfo:
            tripInfo_dict[i['id']] = [float(i['depart']),float(i['arrival'])]
            
        T = args.num_steps                      # total_training_time
        T_round = args.round_duration           # duration of a round
        mu_local_train = args.mu_local_train   # param of shift exponential distribution function for local training delay
        beta_local_train = args.beta_local_train   # param of shift exponential distribution function for local training delay


        num_rounds = int(T/T_round)
        car_calculation_power = {} # 每辆车的计算能力
        cars_dict_per_round = {} # 每轮中曾在道路上行驶的车辆集合
        
        MU_local_train = args.local_iter * args.mu_local_train   # param of shift exponential distribution function for local training delay
        BETA_local_train = args.local_iter * args.beta_local_train   # param of shift exponential distribution function for local training delay
        for k in tripInfo_dict.keys():
            car_calculation_power[k] = args.local_iter / (float(s.exponential(BETA_local_train,(1,)))+MU_local_train)
            
        for r in range(num_rounds):
            cars_dict_per_round[r] = []
        for k,v in tripInfo_dict.items():
            r1 = int(v[0]/T_round)
            r2 = int(v[1]/T_round)
            for r in range(r1,r2+1):
                if r in cars_dict_per_round.keys():
                    cars_dict_per_round[r].append((k,v))
                    
        self.car_calculation_power = car_calculation_power
        self.cars_dict_per_round = cars_dict_per_round
        self.tripInfo_dict = tripInfo_dict
        self.cars_on_road = []
        self.FL_table = {self.round:{}}
        self.car_last_out = None
        self.link = None
        
    def run_a_step(self):
        self.t += self.dt
        if self.link != None and self.link.link_state == "active":
            self.link.update_state(dt=self.dt)

        if self.car_last_out != None:
            self.car_last_out.car_last_update_state(dt=self.dt)
        
        for idx,car in enumerate(self.cars_on_road):
            model_message = car.update_state(dt=self.dt) # ["model_1",{"car_1":13.3, "car_2":3.2}]
            if model_message != None:
                model_name, training_message = model_message
                self.FL_table[self.round][model_name] = training_message
            if car.position > self.road_len:
                self.car_last_out = self.cars_on_road.pop(idx)

        if self.car_last_out!=None and len(self.cars_on_road) > 0 and self.cars_on_road[0].position > self.car_last_out.position-self.V2V_range and self.car_last_out.car_state=="call_v2v":
            self.car_last_out.car_state = "v2v_ing"
            self.link = Link_v2v_3(args=self.args, transmitter=self.car_last_out, 
                                receiver=self.cars_on_road[0], 
                                model_message = self.car_last_out.send_model_message())
        
                    
        # 加入新到达的车辆
        new_car_id = "car_"+str(self.cars_passed)
        if new_car_id in self.tripInfo_dict.keys() and  self.tripInfo_dict[new_car_id][0] <= self.t:
            cal_power = self.car_calculation_power[new_car_id]
            t_in = self.tripInfo_dict[new_car_id][0]
            t_out = self.tripInfo_dict[new_car_id][1]
            self.cars_on_road.append(Car_v2v_3(args=self.args, car_id=new_car_id,cal_power=cal_power, t_in=t_in, t_out=t_out, t=self.t, road_len=self.road_len))
            self.cars_passed += 1
            
        # 判断是否到新的一轮
        if self.t > (self.round + 1) * self.args.round_duration:
            self.round += 1
            self.FL_table[self.round]={}
            for idx,car in enumerate(self.cars_on_road):
                car.reset()
                self.link = None


class Car_v2v_4:
    def __init__(self, args, car_id, cal_power, t_in, t_out,t, road_len=400):
        self.args = args
        self.car_id = car_id
        self.cal_power = cal_power
        self.t_in = t_in
        self.t_out = t_out
        self.speed = road_len / (t_out - t_in)
        self.road_len = road_len
        self.t = t
        self.position = (self.t - self.t_in) * self.speed
        if self.position < 0 or self.position > road_len:
            print("BUG: CAR OUT OF ROAD")
        self.train_list = []
        self.car_state = "download"
        self.car_progress  = 0
    
    def reset(self):
        self.train_list = []
        self.car_state = "download"
        self.car_progress  = 0
        
    def __repr__(self):
        return str((self.car_id, self.position))
        
    
    def update_state(self, dt=0.05):
        self.t += dt
        self.position += self.speed*dt
        
        if self.car_state == "download":
            self.car_progress += dt/self.args.delay_download
            if self.car_progress >= 1:
                self.car_state = "train"
                self.train_list.append(["model_"+str(self.car_id),{self.car_id:0}]) # ["model_1",{"car_1":13.3, "car_2":3.2}]
        elif self.car_state == "train":
            try:
                if self.car_id in self.train_list[0][1].keys():
                    self.train_list[0][1][self.car_id] += self.cal_power*dt
                else:
                    self.car_state = "upload"
                if sum([int(car_iter) for car_iter in self.train_list[0][1].values()]) >= self.args.local_iter:
                    self.car_state = "upload"
            except TypeError:
                import ipdb;ipdb.set_trace()
        elif self.car_state == "upload":
            self.car_progress += dt/self.args.delay_upload
            if self.car_progress >= 1:
                self.car_state = "idle"
                return self.train_list.pop(0) # ["model_1",{"car_1":13.3, "car_2":3.2}]
        elif self.car_state == "idle":
            if len(self.train_list) > 0:
                self.car_state = "train"
        else:
            exit("BUG")
        return None

    def car_last_update_state(self, dt=0.05):
        self.t += dt
        self.position += self.speed*dt
        
        if len(self.train_list) == 0:
            self.car_state = "idle"
        elif self.car_state == "download":
            self.car_state = "call_v2v"
        elif self.car_state == "train":
            if self.car_id in self.train_list[0][1].keys():
                self.train_list[0][1][self.car_id] += self.cal_power*dt
            else:
                self.car_state = "call_v2v"
            if sum([int(car_iter) for car_iter in self.train_list[0][1].values()]) >= self.args.local_iter:
                self.car_state = "call_v2v"
        elif self.car_state == "upload":
            self.car_state = "call_v2v"
        elif self.car_state == "idle":
            self.car_state = "train"
        elif self.car_state == "call_v2v":
            self.car_state = "call_v2v"
        elif self.car_state == "v2v_ing":
            self.car_state = "v2v_ing"
        else:
            exit("BUG")
        return None

    def send_model_message(self):
        if len(self.train_list) > 0:
            model_message = self.train_list.pop(0)
            return model_message
        else:
            return None
    
    def receive_model_message(self, model_message):
        self.train_list.append(model_message)
        self.sort_train_list()
        
    def sort_train_list(self):
        def cmp_func(a,b):
            a1 = sum([int(i) for i in a[1].values()])
            b1 = sum([int(i) for i in b[1].values()])
            if a1 < b1:
                return 1
            elif a1 > b1:
                return -1
            else:
                return 0
        self.train_list.sort(key=cmp_to_key(cmp_func), reverse=False)
    
    
class Link_v2v_4:
    def __init__(self, args, transmitter, receiver, model_message):
        self.args = args
        self.transmitter = transmitter
        self.receiver = receiver
        self.message = model_message
        self.link_progress = 0
        self.link_state = "active"
    
    def update_state(self, dt=0.05):
        self.link_progress += dt/self.args.delay_upload
        if self.link_progress >= 1:
            self.receiver.receive_model_message(self.message)
            self.transmitter.car_state = "idle"
            self.link_state = "over"
            #return self.message, self.receiver.car_id
    
class Road_v2v_4:
    def __init__(self, args, road_len, car_tripinfo, V2V_range=400, dt=0.05):
        self.args = args
        self.road_len = road_len
        self.dt = dt
        self.V2V_range = V2V_range
        self.t = 0 # 当前时间
        self.round = 0 # 当前轮次
        self.cars_passed = 0 # 当前已到达过道路的车辆数
        tripInfo_dict = {}
        for i in car_tripinfo:
            tripInfo_dict[i['id']] = [float(i['depart']),float(i['arrival'])]
        T = args.num_steps                      # total_training_time
        T_round = args.round_duration           # duration of a round
        mu_local_train = args.mu_local_train   # param of shift exponential distribution function for local training delay
        beta_local_train = args.beta_local_train   # param of shift exponential distribution function for local training delay


        num_rounds = int(T/T_round)
        car_calculation_power = {} # 每辆车的计算能力
        cars_dict_per_round = {} # 每轮中曾在道路上行驶的车辆集合
        
        MU_local_train = args.local_iter * args.mu_local_train   # param of shift exponential distribution function for local training delay
        BETA_local_train = args.local_iter * args.beta_local_train   # param of shift exponential distribution function for local training delay
        for k in tripInfo_dict.keys():
            car_calculation_power[k] = args.local_iter / (float(s.exponential(BETA_local_train,(1,)))+MU_local_train)
            
        for r in range(num_rounds):
            cars_dict_per_round[r] = []
        for k,v in tripInfo_dict.items():
            r1 = int(v[0]/T_round)
            r2 = int(v[1]/T_round)
            for r in range(r1,r2+1):
                if r in cars_dict_per_round.keys():
                    cars_dict_per_round[r].append((k,v))
                    
        self.car_calculation_power = car_calculation_power
        self.cars_dict_per_round = cars_dict_per_round
        self.tripInfo_dict = tripInfo_dict
        self.cars_on_road = []
        self.FL_table = {self.round:{}}
        self.cars_out = []
        self.links = []
        
    def run_a_step(self):
        self.t += self.dt
        for link in self.links:
            if link.link_state == "active":
                link.update_state(dt=self.dt)

        for car in self.cars_out:
            car.car_last_update_state(dt=self.dt)
        
        for idx,car in enumerate(self.cars_on_road):
            model_message = car.update_state(dt=self.dt) # ["model_1",{"car_1":13.3, "car_2":3.2}]
            if model_message != None:
                model_name, training_message = model_message
                self.FL_table[self.round][model_name] = training_message
            if car.position > self.road_len:
                self.cars_out.append(self.cars_on_road.pop(idx))

        for idx,car_out in enumerate(self.cars_out):
            car_rec = None
            for car_on in self.cars_on_road:
                if car_on.position > car_out.position-self.V2V_range:
                    car_rec = car_on

            if car_rec!=None and car_out.car_state=="call_v2v":
                car_out.car_state = "v2v_ing"
                self.links.append(Link_v2v_4(args=self.args, transmitter=car_out, 
                              receiver=car_rec, 
                              model_message = car_out.send_model_message())
                              )
                    
        # 加入新到达的车辆
        new_car_id = "car_"+str(self.cars_passed)
        if new_car_id in self.tripInfo_dict.keys() and  self.tripInfo_dict[new_car_id][0] <= self.t:
            cal_power = self.car_calculation_power[new_car_id]
            t_in = self.tripInfo_dict[new_car_id][0]
            t_out = self.tripInfo_dict[new_car_id][1]
            self.cars_on_road.append(Car_v2v_4(args=self.args, car_id=new_car_id,cal_power=cal_power, t_in=t_in, t_out=t_out, t=self.t, road_len=self.road_len))
            self.cars_passed += 1
            
        # 判断是否到新的一轮
        if self.t > (self.round + 1) * self.args.round_duration:
            self.round += 1
            self.FL_table[self.round]={}
            for idx,car in enumerate(self.cars_on_road):
                car.reset()
            self.links = []
            self.cars_out = []





