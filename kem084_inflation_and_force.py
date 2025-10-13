#!/usr/bin/env python3.8
"""Jimstron test example. Do not delete. New files can be create via copying this
 either manual or creating a new test in the GUI"""

import random, yaml, json, datetime, time
import rospy

from std_msgs.msg import Bool
from std_msgs.msg import String
from std_msgs.msg import Float32

from std_srvs.srv import SetBool
from std_srvs.srv import Trigger
from std_srvs.srv import TriggerResponse

from jimstron.srv import SetFloat
from jimstron.srv import SetString

from click_plc_ros.srv import SetRegister
from click_plc_ros.srv import GetRegister

from laumas_ros.srv import SetFloat as SetFloatLaumas

# ROS Image message -> OpenCV2 image converter, OpenCV2 for saving an image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv2

#Globals
test_stop_flag = False

position = None
velocity = None
torque = None

servo_status = None
servo_motion_complete = False

force = None
force_raw = None
gpio_data = None

image_msg = None

STOP_FORCE = 10.0
# WAIT_OFFSET = 150.0
MODEL_EXTENSION = 156
AMPLITUDE = 22
TOLERANCE = 10

##################################################################################
#                                    Callbacks
##################################################################################

def position_callback(msg):
    global position
    position = float(msg.data)

def velocity_callback(msg):
    global velocity
    velocity = float(msg.data)

def torque_callback(msg):
    global torque
    torque = float(msg.data)

def servo_status_callback(msg):
    global servo_status_callback
    servo_status_callback = json.loads(msg.data)

def servo_moving_callback(msg):
    global servo_motion_complete
    servo_motion_complete = bool(msg.data)

def gpio_data_update(msg):
    global gpio_data
    gpio_data = json.loads(msg.data)

def load_cell_raw_update(msg):
    global force_raw 
    force_raw = float(msg.data)

def load_cell_force_update(msg):
    global force, max_pulloff_force
    force = float(msg.data)

def image_callback(msg:CompressedImage):
    global image_msg
    image_msg = msg

def test_stop_callback(msg):
    global test_stop_flag
    test_stop_flag = True
    test_status_pub_.publish("STOPPING")
    res = TriggerResponse()
    res.success = True
    return res

##################################################################################
#                       Publishers, Subscribers, Services
#               * See rostopic list for more individual topics 
##################################################################################

### ROS handlers for the servo motor
rospy.init_node("jimstron_test", log_level=rospy.INFO, anonymous=False, disable_signals=True)

#Test
test_status_pub_ =  rospy.Publisher("test/status", String, queue_size=5, latch=True)
rospy.Service("test/stop", Trigger, test_stop_callback)

#Logger
logging_start_ = rospy.ServiceProxy("logger/start", Trigger)
logging_stop_ = rospy.ServiceProxy("logger/stop", Trigger)
logging_save_ = rospy.ServiceProxy("logger/save", SetString)

# Servo - Subscribers
rospy.Subscriber("jimstron/position", Float32, position_callback)
rospy.Subscriber("jimstron/velocity", Float32, velocity_callback)
rospy.Subscriber("jimstron/torque", Float32, torque_callback)
rospy.Subscriber("servo/status/all", String, servo_status_callback)
rospy.Subscriber("servo/status/move_cmd_complete", Bool, servo_moving_callback) 
# Servo - Services
servo_set_acc_lim_ = rospy.ServiceProxy("servo/set_acc_lim", SetFloat)
servo_set_vel_lim_ = rospy.ServiceProxy("servo/set_vel_lim", SetFloat)
servo_set_decel_lim_ = rospy.ServiceProxy("servo/set_decel_lim", SetFloat)
servo_absolute_move_ = rospy.ServiceProxy("servo/absolute_move", SetFloat)
servo_relative_move_ = rospy.ServiceProxy("servo/relative_move", SetFloat)
servo_velocity_move_ = rospy.ServiceProxy("servo/velocity_move", SetFloat)
servo_reset_ = rospy.ServiceProxy("servo/reset", Trigger)
servo_enable_ = rospy.ServiceProxy("servo/enable", SetBool)
servo_stop_ = rospy.ServiceProxy("servo/stop", Trigger)
servo_cancel_ = rospy.ServiceProxy("servo/cancel", Trigger)
servo_home_ = rospy.ServiceProxy("servo/home", Trigger)

#Load cell
rospy.Subscriber("load_cell/raw", Float32, load_cell_raw_update)
rospy.Subscriber("load_cell/force", Float32, load_cell_force_update)
load_cell_zero_ = rospy.ServiceProxy("load_cell/zero", Trigger)
load_cell_calibrate_ = rospy.ServiceProxy("load_cell/calibrate", SetFloatLaumas)

#GPIO
rospy.Subscriber("plc/all_data", String, gpio_data_update)
plc_set_ = rospy.ServiceProxy("plc/set", SetRegister)
plc_get_ = rospy.ServiceProxy("plc/get", GetRegister)

#Camera
# camera_1_save_image = rospy.ServiceProxy("camera_1/save_image", SetString)
# camera_1_start_timelapse = rospy.ServiceProxy("camera_1/timelapse/start", SetString)
# camera_1_stop_timelapse = rospy.ServiceProxy("camera_1/timelapse/stop", Trigger)
# camera_2_save_image = rospy.ServiceProxy("camera_2/save_image", SetString)
# camera_2_start_timelapse = rospy.ServiceProxy("camera_2/timelapse/start", SetString)
# camera_2_stop_timelapse = rospy.ServiceProxy("camera_2/timelapse/stop", Trigger)

##################################################################################
#                                Helper functions 
##################################################################################

def ros_msg_to_json(msg):
    y = yaml.load(str(msg), yaml.SafeLoader)
    return json.loads(json.dumps(y))

# def save_image(dir, name=None, extension=".jpeg"):
#     global image_msg
#     if not os.path.isdir(dir):
#         os.makedirs(dir)
        
#     if image_msg:
#         filename = name if name else str(image_msg.header.stamp)
#         subscribed_image = CvBridge().compressed_imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
#         cv2.imwrite(os.path.join(dir, filename+extension), subscribed_image)
#         image_msg = None  #Clear image temp to stop the same img getting saved next call
#     else:
#         rospy.logwarn("No image ready to save")

def wait(break_func=None, *args):
    rospy.sleep(0.1)    # Avoid race condition, really small moves might be effected by this TODO: remove need for this
    rate = rospy.Rate(50)
    while not servo_motion_complete:
        # Check if break function exists. Stop servo and return on True
        if break_func and break_func(*args): 
            servo_stop_()
            return False # Wait exited early
        rate.sleep()
    return True # Wait completed succesfully

def abs_limit_force(force_limit):
    if abs(force) >= force_limit:
        rospy.logwarn("Force limit exceeded!") 
        return True
    return False



# log an event with the time from `test_start_time`.
def log_event(msg):
    """
    Print/log a message that includes the time (in seconds) from test_start_time.
    """
    global test_start_time
    if test_start_time is None:
        rospy.logwarn("test_start_time not set! Cannot log time offset.")
        rospy.loginfo(msg)
        return

    elapsed = rospy.Time.now() - test_start_time
    rospy.loginfo(f"[{elapsed.to_sec():.2f}s] {msg}")

def find_model_position(force_threshold=3.0, max_travel=500.0, velocity_lim=20.0, velocity=10.0):
    """
    Move crosshead downward slowly until force >= force_threshold.
    Then store that position in global model_position.

    - force_threshold: minimal force at which we say "the model is engaged."
    - max_travel: how far in mm we can move down before giving up.

    We do a velocity_move downwards (like -1 mm/s) and watch force or distance.
    """
    global model_position
    model_position = None

    # Set (slow) velocity limit
    servo_set_vel_lim_(velocity_lim)
    rospy.sleep(0.2)

    # Move downward with until force >= threshold
    servo_velocity_move_(velocity)

    rate = rospy.Rate(50)
    start_pos = position if position else 0.0
    while not rospy.is_shutdown():
        if test_stop_flag:
            servo_stop_()
            return

        # check force
        if force is not None and abs(force) >= force_threshold:
            # we found the model
            servo_stop_()
            model_position = position
            log_event(f"Found model at position={model_position:.2f} mm, force={force:.2f} N")
            return

        # check how far we've traveled
        curr_pos = position if position else 0.0
        if abs(curr_pos - start_pos) > max_travel:
            servo_stop_()
            rospy.logwarn("Max travel reached without detecting the model.")
            return

        rate.sleep()
        
def force_capture(force_threshold=3.0, max_travel=50.0, velocity_lim=5.0, velocity=5.0,max_seconds=3.0):
    """
    Move crosshead downward slowly until force >= force_threshold.
    Then store that position in global model_position.

    - force_threshold: minimal force at which we say "the model is engaged."
    - max_travel: how far in mm we can move down before giving up.

    We do a velocity_move downwards (like -1 mm/s) and watch force or distance.
    """
    global model_position
    model_position = None
    i = 0
    bool_thing = 0

    # Set (slow) velocity limit
    servo_set_vel_lim_(velocity_lim)
    rospy.sleep(0.2)

    # Move downward with until force >= threshold
    servo_velocity_move_(velocity)

    rate_value = 4

    rate = rospy.Rate(rate_value)
    start_pos = position if position else 0.0
    while not rospy.is_shutdown():
        i = i + 1
        # if i % 2 == 0:
        #     bool_thing = True
        # else
        #     bool_thing = False
        #     valve_on()
            
        # pulse inflation
        valve_on()
        
        if test_stop_flag:
            servo_stop_()
            return

        # check force
        if (force is not None and abs(force) >= force_threshold) or (i > rate_value*max_seconds):
            # we found the model
            servo_stop_()
            valve_off()
            model_position = position
            # log_event(f"Found model at position={model_position:.2f} mm, force={force:.2f} N")
            return

        # check how far we've traveled
        curr_pos = position if position else 0.0
        if abs(curr_pos - start_pos) > max_travel:
            servo_stop_()
            rospy.logwarn("Max travel reached without detecting the model.")
            return

        rate.sleep()
        valve_off()
        

def valve_on():
    """
    Turn on pneumatic valve using PLC output 'C1'.
    Adjust the PLC register name if needed.
    """
    if gpio_data:
        plc_set_("C2", [True])
        # rospy.loginfo("Valve turned ON")
        # log_event("Valve turned ON")
    else:
        rospy.logwarn("No gpio_data available, cannot set valve")


def valve_off():
    """
    Turn off pneumatic valve using PLC output 'C1'.
    """
    if gpio_data:
        plc_set_("C2", [False])
        # rospy.loginfo("Valve turned OFF")
        # log_event("Valve turned OFF")
    else:
        rospy.logwarn("No gpio_data available, cannot set valve")

def negative_valve_on():
    """
    Turn on pneumatic valve using PLC output 'C2'.
    Adjust the PLC register name if needed.
    """
    if gpio_data:
        plc_set_("C1", [True])
        # rospy.loginfo("Valve turned ON")
        # log_event("Valve turned ON")
    else:
        rospy.logwarn("No gpio_data available, cannot set valve")

def negative_valve_off():
    """
    Turn off pneumatic valve using PLC output 'C2'.
    """
    if gpio_data:
        plc_set_("C1", [False])
        # rospy.loginfo("Valve turned OFF")
        # log_event("Valve turned OFF")
    else:
        rospy.logwarn("No gpio_data available, cannot set valve")
        
def neutral_valve_on():
    """
    Turn on pneumatic valve using PLC output 'C3'.
    Adjust the PLC register name if needed.
    """
    if gpio_data:
        plc_set_("C3", [True])
        # rospy.loginfo("Valve turned ON")
        # log_event("Valve turned ON")
    else:
        rospy.logwarn("No gpio_data available, cannot set valve")

def neutral_valve_off():
    """
    Turn off pneumatic valve using PLC output 'C3'.
    """
    if gpio_data:
        plc_set_("C3", [False])
        # rospy.loginfo("Valve turned OFF")
        # log_event("Valve turned OFF")
    else:
        rospy.logwarn("No gpio_data available, cannot set valve")
        
##################################################################################
#                                ADD TEST CODE HERE 
##################################################################################

def pre_test():
    """
    1) Zero load cell 
    2) Home the Instron
    3) Start camera timelapse
    4) Record test_start_time
    5) Find the unactuated model position (small downward search)
    """
    global test_start_time
    global model_position

    test_status_pub_.publish("PRE-TEST")
    rospy.loginfo("Zeroing load cell ...")
    load_cell_zero_()
    rospy.sleep(1.0)

    rospy.loginfo("Homing the crosshead ...")
    servo_set_vel_lim_(20.0)
    # servo_home_()
    servo_absolute_move_(-370.0)
    if not wait(abs_limit_force, STOP_FORCE):
        return
    # wait(abs_limit_force, STOP_FORCE)
    # Do i need to set servo_home as zero for future absolute moves?
    rospy.sleep(2.0)
    
    # record the test start time
    test_start_time = rospy.Time.now()
    rospy.loginfo("test_start_time recorded.")
    # rospy.loginfo(f"test start time: {test_start_time}")

def test():
    """
    Main test loop. 
    - Set crosshead velocity
    - Move crosshead down
    - Turn on valve, wait a bit
    - Check force
    - Turn off valve
    - Move crosshead back
    """
    test_status_pub_.publish("RUNNING")
    
    # find the unactuated model position
    find_model_position(force_threshold=2.0, max_travel=500.0)
    rospy.sleep(1)
    if model_position is not None:
        rospy.loginfo(f"Model position set to {model_position:.2f} mm")
        test_status_pub_.publish("Model Found")
        # rospy.sleep(2)
    else:
        rospy.logwarn("Could not detect model. Possibly no force threshold reached.")
        

    rospy.loginfo("Pre-testing Complete")
    
    rospy.sleep(1)

    # Set velocity limit to: 10 mm/s
    servo_set_vel_lim_(10.0)

    # Move to test position if set
    if model_position is not None:
        # Move to test position
        test_position = model_position + MODEL_EXTENSION - AMPLITUDE + TOLERANCE
        rospy.loginfo(f"Moving to test/end-effector position: {test_position:.2f} mm ...")
        servo_absolute_move_(test_position)
        if not wait(abs_limit_force, 25):
            return
        rospy.loginfo(f"Crosshead waiting at position: {position:.2f} mm ...")
        test_status_pub_.publish("at test position")
    else:
        rospy.logwarn("Model position not set. Attempting relative move.")
        # Move absolute to -50 mm from home
        rospy.loginfo("Moving crosshead + model_extension ...")
        servo_relative_move_(model_extension)
        if not wait(abs_limit_force, STOP_FORCE): 
            return
        
    
    test_status_pub_.publish("At Test Position")
    rospy.sleep(2)
    
    # neutral_valve_off()
    plc_set_("C3", [False])
    test_status_pub_.publish("Atmosphere OFF")
    rospy.sleep(1)

    # Turn the valve on for inflation
    valve_on()
    test_status_pub_.publish("Valve ON")
    rospy.sleep(2)  # hold for 2 seconds, see how force changes

    # Check force
    # if force is not None:
    #     rospy.loginfo(f"Current force reading: {force:.2f} N")

    # Turn valve off
    valve_off()
    test_status_pub_.publish("Valve OFF")
    rospy.sleep(1.0)
    
    # # move to model top 
    # if model_position is not None:
    #     # Move to test position
    #     test_position = model_position + MODEL_EXTENSION - AMPLITUDE
    #     rospy.loginfo(f"Moving to test/end-effector position: {test_position:.2f} mm ...")
    #     servo_absolute_move_(test_position)
    #     if not wait(abs_limit_force, STOP_FORCE):
    #         return
    #     rospy.loginfo(f"Crosshead waiting at position: {position:.2f} mm ...")
        
    # Find Model end-affector/tip
    find_model_position(force_threshold=1.0, max_travel=80.0)
        
    # pulse inflate model for force capture
    force_capture(force_threshold=5.0)
        
    test_status_pub_.publish("Found Force")
    rospy.sleep(2)
    
    # # servo_absolute_move_(MODEL_EXTENSION+AMPLITDE+TOLERANCE)
    
    # # move for vacuum test cycle
    # # if model_position is not None:
    # #     # Move to test position
    # #     test_position = model_position + MODEL_EXTENSION - AMPLITUDE + TOLERANCE
    # #     rospy.loginfo(f"Moving to test/end-effector position: {test_position:.2f} mm ...")
    # #     servo_absolute_move_(test_position)
    # #     if not wait(abs_limit_force, STOP_FORCE):
    # #         return
    # #     rospy.loginfo(f"Crosshead waiting at position: {position:.2f} mm ...")
    # # else:
    # #     rospy.logwarn("Model position not set. Attempting relative move.")
    # #     # Move absolute to -50 mm from home
    # #     rospy.loginfo("Moving crosshead + model_extension ...")
    # #     servo_relative_move_(model_extension)
    # #     if not wait(abs_limit_force, STOP_FORCE): 
    # #         return
    
    servo_absolute_move_(test_position)
    if not wait(abs_limit_force, STOP_FORCE): 
        return
    
    test_status_pub_.publish("At Test Position")
    
    # Turn neutral valve on and off
    neutral_valve_off()
    test_status_pub_.publish("Atmosphere ON")
    rospy.sleep(1)
    
    # Turn the valve on for inflation
    valve_on()
    test_status_pub_.publish("Valve ON")
    rospy.sleep(0.5)  # hold for 3 seconds, see how force changes

    # Turn valve off
    valve_off()
    test_status_pub_.publish("Valve OFF")
    rospy.sleep(1.0)
    
    # Turn negative valve on
    negative_valve_on()
    test_status_pub_.publish("Negative ON")
    rospy.sleep(3)
    
    # Turn negative valve off
    negative_valve_off()
    test_status_pub_.publish("Negative OFF")
    rospy.sleep(1)
    
    # Turn neutral valve on and off
    neutral_valve_off()
    test_status_pub_.publish("Atmosphere OFF")
    rospy.sleep(1)
    
        # Turn neutral valve on and off
    neutral_valve_on()
    test_status_pub_.publish("Atmosphere ON")
    rospy.sleep(1)
    
    # done


def post_test():
    """
    Wrap up:
    - Move crosshead to home (0 mm)
    - Stop camera timelapse
    - Save logs
    """
    test_status_pub_.publish("POST-TEST")
    
    # # Save logging data with timestamp
    # timestamp = datetime.datetime.now().strftime("%Y-%m-%d_T%H%M%S")
    # logging_save_(f"jimstron_data_{timestamp}")
    # rospy.sleep(2)

    # rospy.loginfo("Moving crosshead back to home (0 mm) ...")
    # servo_absolute_move_(0.0)
    # wait(abs_limit_force, STOP_FORCE)
    rospy.sleep(2)
    

    # # # Stop camera timelapse
    # # camera_1_stop_timelapse()
    # # rospy.loginfo("Camera timelapse stopped.")



##################################################################################
#
##################################################################################

if __name__ == "__main__":
    try:
        rospy.sleep(0.5)  # Wait for node subscribers to subscribe 

        # rospy.loginfo("Test Started!")
        test_status_pub_.publish("PRE-TEST")
        # logging_start_()
        pre_test()
  
        # test_status_pub_.publish("RUNNING") 
        rospy.wait_for_service('logger/start') # Should be running but to make sure
        
        logging_start_()
        test()
        logging_stop_()
        
        test_status_pub_.publish("POST-TEST")
        post_test()

        if (test_stop_flag): rospy.logwarn("Test stopped by user")
        test_status_pub_.publish("COMPLETE")
        rospy.loginfo("Test Complete!")

    except rospy.ROSInterruptException:
        rospy.logerror("Test Failed!")
