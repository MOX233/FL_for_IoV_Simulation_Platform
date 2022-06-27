#!/usr/bin/env python
import os
import sys

import csv
import random
from decimal import Decimal


def generate_routefile(args, save_dir="./sumo_data"):
    random.seed(args.seed)  # make tests reproducible
    
    num_steps = args.num_steps  # number of time steps
    Lambda = args.Lambda       # arrival rate of car flow
    accel = args.accel         # accelerate of car flow
    decel = args.decel         # decelerate of car flow
    sigma = args.sigma         # imperfection of drivers, which takes value on [0,1], with 0 meaning perfection and 1 meaning imperfection
    carLength = args.carLength # length of cars
    minGap = args.minGap       # minimum interval between adjacent cars
    maxSpeed = args.maxSpeed   # maxSpeed for cars
    speedFactoer_mean = args.speedFactoer_mean 
    speedFactoer_dev = args.speedFactoer_dev
    speedFactoer_min = args.speedFactoer_min
    speedFactoer_max = args.speedFactoer_max
    
    speedFactoer = "normc({mean}, {dev}, {min}, {max})".format(**{
        "mean":speedFactoer_mean,
        "dev":speedFactoer_dev,
        "min":speedFactoer_min,
        "max":speedFactoer_max,
    }) # can be given as "norm(mean, dev)" or "normc(mean, dev, min, max)"

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir,"road.rou.xml"), "w") as routes:
        print("""<routes>
        <vType id="typecar" accel="{accel}" decel="{decel}" sigma="{sigma}" length="{carLength}" minGap="{minGap}" maxSpeed="{maxSpeed}" speedFactoer="{speedFactoer}" guiShape="passenger"/>
        <route id="right" edges="right1 right12 right2" />
        <route id="left" edges="left2 left21 left1" />""".format(**{
            "accel":accel,
            "decel":decel,
            "sigma":sigma,
            "carLength":carLength,
            "minGap":minGap,
            "maxSpeed":maxSpeed,
            "speedFactoer":speedFactoer,
        }), file=routes)
        vehNr = 0
        for i in range(num_steps):
            # just right traffic
            if random.uniform(0, 1) < Lambda:
                print('    <vehicle id="car_%i" type="typecar" route="right" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
        print("</routes>", file=routes)


def generate_netfile(args, save_dir="./sumo_data"):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir,"road.net.xml"), "w") as netfile:
        print("""<?xml version="1.0" encoding="UTF-8"?>
<net version="1.9" junctionCornerDetail="5" limitTurnSpeed="5.50" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">
    <location netOffset="0.00,0.00" convBoundary="-210.00,0.00,210.00,0.00" origBoundary="-10000000000.00,-10000000000.00,10000000000.00,10000000000.00" projParameter="!"/>
    <edge id="left1" from="entrance1" to="end1" priority="-1" length="10.00">
        <lane id="left1_0" index="0" speed="{:.2f}" length="10.00" shape="-200.00,1.60 -210.00,1.60"/>
    </edge>
    <edge id="left2" from="end2" to="entrance2" priority="-1" length="10.00">
        <lane id="left2_0" index="0" speed="{:.2f}" length="10.00" shape="210.00,1.60 200.00,1.60"/>
    </edge>
    <edge id="left21" from="entrance2" to="entrance1" priority="-1" length="400.00">
        <lane id="left21_0" index="0" speed="{:.2f}" length="400.00" shape="200.00,1.60 -200.00,1.60"/>
    </edge>
    <edge id="right1" from="end1" to="entrance1" priority="-1" length="10.00">
        <lane id="right1_0" index="0" speed="{:.2f}" length="10.00" shape="-210.00,-1.60 -200.00,-1.60"/>
    </edge>
    <edge id="right12" from="entrance1" to="entrance2" priority="-1" length="400.00">
        <lane id="right12_0" index="0" speed="{:.2f}" length="400.00" shape="-200.00,-1.60 200.00,-1.60"/>
    </edge>
    <edge id="right2" from="entrance2" to="end2" priority="-1" length="10.00">
        <lane id="right2_0" index="0" speed="{:.2f}" length="10.00" shape="200.00,-1.60 210.00,-1.60"/>
    </edge>
    <junction id="end1" type="dead_end" x="-210.00" y="0.00" incLanes="left1_0" intLanes="" shape="-210.00,0.00 -210.00,3.20 -210.00,0.00"/>
    <junction id="end2" type="dead_end" x="210.00" y="0.00" incLanes="right2_0" intLanes="" shape="210.00,0.00 210.00,-3.20 210.00,0.00"/>
    <junction id="entrance1" type="priority" x="-200.00" y="0.00" incLanes="left21_0 right1_0" intLanes="" shape="-200.00,3.20 -200.00,-3.20 -200.00,3.20">
        <request index="0" response="00" foes="00"/>
        <request index="1" response="00" foes="00"/>
    </junction>
    <junction id="entrance2" type="priority" x="200.00" y="0.00" incLanes="left2_0 right12_0" intLanes="" shape="200.00,3.20 200.00,-3.20 200.00,3.20">
        <request index="0" response="00" foes="00"/>
        <request index="1" response="00" foes="00"/>
    </junction>
    <connection from="left2" to="left21" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from="left21" to="left1" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from="right1" to="right12" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from="right12" to="right2" fromLane="0" toLane="0" dir="s" state="M"/>
</net>""".format(args.maxSpeed,args.maxSpeed,args.maxSpeed,args.maxSpeed,args.maxSpeed,args.maxSpeed), file=netfile)

def generate_cfgfile(args, save_dir="./sumo_data"):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir,"road.sumocfg"), "w") as cfgfile:
        print("""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="road.net.xml"/>
        <route-files value="road.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
    </time>
    <report>
        <verbose value="true"/>
        <no-step-log value="true"/>
    </report>
</configuration>""", file=cfgfile)

def sumo_run(args, save_dir="./sumo_data"):
    # we need to import python modules from the $SUMO_HOME/tools directory
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
        import traci  # noqa
        from sumolib import checkBinary  # noqa
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")
    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run

    sumoBinary = checkBinary('sumo')

    # first, generate the route file for this simulation
    generate_routefile(args,save_dir=save_dir)
    generate_netfile(args,save_dir=save_dir)
    generate_cfgfile(args,save_dir=save_dir)
    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    traci.start([sumoBinary, "-c", os.path.join(save_dir, "road.sumocfg"),
                             "--tripinfo-output", os.path.join(save_dir, "tripinfo.xml"),
                             '--step-length', str(args.step_length),])

    """execute the TraCI control loop"""
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1
    traci.close()
    sys.stdout.flush()

def sumo_run_with_trajectoryInfo(args, save_dir="./sumo_data"):
    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run

    # we need to import python modules from the $SUMO_HOME/tools directory
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
        import traci  # noqa
        from sumolib import checkBinary  # noqa
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    if args.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # first, generate the route file for this simulation
    generate_routefile(args,save_dir=save_dir)
    generate_netfile(args,save_dir=save_dir)
    generate_cfgfile(args,save_dir=save_dir)
    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    traci.start([sumoBinary, "-c", os.path.join(save_dir, "road.sumocfg"),
                             "--tripinfo-output", os.path.join(save_dir, "tripinfo.xml"),
                             '--step-length', str(args.step_length),])

    """execute the TraCI control loop"""
    step = 0
    w = csv.writer(open(args.trajectoryInfo_path, 'w',newline=""))
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1
        for veh_i, veh in enumerate(traci.vehicle.getIDList()):
            (x, y),speed, = [f(veh) for f in [
                traci.vehicle.getPosition,
                traci.vehicle.getSpeed, #Returns the speed of the named vehicle within the last step [m/s]; error value: -1001

            ]]
            w.writerow([step,veh,veh_i,x,speed])
    traci.close()
    sys.stdout.flush()


def read_tripInfo(tripInfo_path = 'tripinfo.xml'):
    pass
    car_tripinfo = []
    with open(tripInfo_path, 'r') as f:
        tripinfo = f.readlines()
        for line in tripinfo:
            if line.startswith('    <tripinfo'):
                car_info = line[14:-3].split(" ")
                car_dict = dict()
                for item in car_info:
                    key,value = item.split('=')
                    car_dict[key] = value[1:-1]
                car_tripinfo.append(car_dict)
    return car_tripinfo


def read_trajectoryInfo(args):
    r = csv.reader(open(args.trajectoryInfo_path,'r'))
    car_trajectory = {}
    for row in r:
        [step,veh,veh_i,x,speed] = row
        step,x,speed = int(step),float(x),float(speed)
        timeslot = float(step*Decimal(str(args.step_length)))
        if not veh in car_trajectory:
            car_trajectory[veh] = []
            car_trajectory[veh].append({
                'timeslot':timeslot,
                'position':x,
                'speed':speed,
            })
        else:
            car_trajectory[veh].append({
                'timeslot':timeslot,
                'position':x,
                'speed':speed,
            })
    return car_trajectory