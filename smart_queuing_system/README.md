# Smart Queuing System
This is the second project for Intel Edge AI for IoT developers nanodegree. In this project we are given three scenarios-
- Manufacturing
- Retail
- Transportation

These scenarios come from three different industry sectors which try to mimic the requirements of a real client. Our job will be to analyze the constraints, propose the hardware solution and build out and test the applicationon the [Intel DEVCloud](https://devcloud.intel.com/edge/). Specifically the goal is to build an application which reduces congestion and queuing systems, situations where the queues are not managed efficiently- one queue may be overloaded while another is empty. Along with the scenarios, sample video footage from the client is provided which can be used to test our applications' performance.

## Goal of the project
Reduce the congestion in the queues using Intel's OpenVINO API and the person detection model from the Open Model Zoo to count the number of people in each queue, so that people can be directed to the least congested queue. We need to build an application that finds the optimized solution for each scenario which satisfy the client's requirements like the limitations on the budgets they can spend on the hardware and the power consumption requirements for the hardware.

## Main Tasks
- Propose a possible hardware solution
- Build out the application and test its performance on the DevCloud using multiple hardware types
- Compare the performance to see which hardware performed best
- Validate the proposal based on the test results

## Scenario 1: Manufacturing
> __Which hardware might be most appropriate for this scenario? (CPU/IGPU/VPU/FPGA)__
>
> Proposed Hardware: FPGA

| Requirements observed | How does the chosen hardware meet this requirement? |
| --- | --- |
| Client requirement is to do image processing 5 times per second on a video feed of 30-35 FPS | FPGA provides low latency as compared to other devices |
| It can be easily reprogrammed and optimized for other tasks like finding flaws in the semiconductor chips | FPGAs are ideal for this scenario because they are very flexible in the sense that they are field programmable |
| It should at least last for 5-10 years | FPGAs that use devices from Intel’s IoT group have a guaranteed availability of 10 years, from start to production | 
| Budget is not a constraint since the revenue is good | FPGAs are at least $1000 |

### Queue Monitoring Requirements:
| Maximum number of people in the queue | 5 |
| --- | --- |
| __Model precision chosen (FP32, FP16, or INT8)__ | __FP16__ |

### Test Results

All the scenarios are benchmarked for three metrics across different hardware - Model load time, Inference time and Frames per second.
- __Model load time__

![model_load_time_manufacturing](images/model_load_time_manufacturing.png)

- __Inference time__

![inference_time_manufacturing](images/inference_time_manufacturing.png)

- __Frames per second__

![fps_manufacturing](images/fps_manufacturing.png)

### Final Hardware Recommendation
> As per the requirements listed above, __FPGA__ indeed proves out to be the best hardware for this scenario since it offers the highest frame rate and lowest inference time.