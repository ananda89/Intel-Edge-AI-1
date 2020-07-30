
import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys


class Queue:
    '''
    Class for dealing with queues
    '''
    def __init__(self):
        self.queues=[]

    def add_queue(self, points):
        self.queues.append(points)

    def get_queues(self, image):
        for q in self.queues:
            x_min, y_min, x_max, y_max=q
            frame=image[y_min:y_max, x_min:x_max]
            yield frame
    
    def check_coords(self, coords):
        d={k+1:0 for k in range(len(self.queues))}
        for coord in coords:
            for i, q in enumerate(self.queues):
                if coord[0]>q[0] and coord[2]<q[2]:
                    d[i+1]+=1
        return d


class PersonDetect:
    '''
    Class for the Person Detection Model.
    '''

    def __init__(self, model_name, device, threshold=0.60):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold

        try:
            self.core = IECore()
            self.model = self.core.read_network(self.model_structure, self.model_weights) # use the latest version to avoid deprecation warnings
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
        '''
        TODO: This method needs to be completed by you
        '''
        self.net = self.core.load_network(self.model, device_name=self.device, num_requests=1)
        
    def predict(self, image, initial_w, initial_h):
        '''
        TODO: This method needs to be completed by you
        '''
        processed_image = self.preprocess_input(image)
        input_dict = {self.input_name:processed_image}
        output = self.net.infer(input_dict)
        coordinates, probas = self.preprocess_outputs(output[self.output_name], initial_w, initial_h)
        self.draw_outputs(coordinates, probas, image)
        return coordinates, image
    
    def draw_outputs(self, coords, probas, image):
        '''
        TODO: This method needs to be completed by you
        '''
        font = cv2.FONT_HERSHEY_SIMPLEX
        color_bbox = (32,32,32)
        color_text =(60,20,220)
        for c, prob in zip(coords, probas):
            cv2.rectangle(image, (c[0], c[1]), (c[2], c[3]), color_bbox, 3)
            text = 'human: {:.2f}%'.format(prob)
            cv2.putText(image, text, (c[0]+2, c[1]+20), font,
                            fontScale=0.99, thickness=2, color=color_text)

    def preprocess_outputs(self, outputs, width, height):
        '''
        TODO: This method needs to be completed by you
        '''
        coordinates = []
        probas = []
        for box in outputs[0][0]:  # Output shape is 1x1x100x7
            conf = box[2]
            if (conf >= self.threshold):
                prob = conf
                xmin = int(box[3]*width)
                ymin = int(box[4]*height)
                xmax = int(box[5]*width)
                ymax = int(box[6]*height)
                coordinates.append((xmin, ymin, xmax, ymax))
                probas.append(prob*100)
#                 print(prob*100)
        return coordinates, probas

    def preprocess_input(self, image):
        '''
        TODO: This method needs to be completed by you
        '''
        processed_image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        processed_image = processed_image.transpose((2, 0, 1))
        processed_image = processed_image.reshape(1, *processed_image.shape)
        
        return processed_image


def main(args):
    model=args.model
    device=args.device
    video_file=args.video
    max_people=args.max_people
    threshold=args.threshold
    output_path=args.output_path

    start_model_load_time=time.time()
    pd = PersonDetect(model, device, threshold)
    pd.load_model()
    total_model_load_time = time.time() - start_model_load_time

    queue=Queue()
    
    try:
        queue_param=np.load(args.queue_param)
        for q in queue_param:
            queue.add_queue(q)
    except:
        print("error loading queue param file")

    try:
        cap=cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: "+ video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)
    
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)
    
    counter=0
    start_inference_time=time.time()

    try:
        while cap.isOpened():
            ret, frame=cap.read()
            if not ret:
                break
            counter+=1
            
            coords, image= pd.predict(frame, initial_w, initial_h)
            num_people= queue.check_coords(coords)
            print(f"Total People in frame = {len(coords)}")
            print(f"Number of people in queue = {num_people}")
            out_text=""
            y_pixel=25
            
            for k, v in num_people.items():
                out_text += f"No. of People in Queue {k} is {v} "
                if v >= int(max_people):
                    out_text += f" Queue full; Please move to next Queue "
                cv2.putText(image, out_text, (15, y_pixel), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                out_text=""
                y_pixel+=40
            out_video.write(image)
            
        total_time=time.time()-start_inference_time
        total_inference_time=round(total_time, 1)
        fps=counter/total_inference_time

        with open(os.path.join(output_path, 'stats.txt'), 'w') as f:
            f.write(str(total_inference_time)+'\n')
            f.write(str(fps)+'\n')
            f.write(str(total_model_load_time)+'\n')

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run Inference: ", e)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default=None)
    parser.add_argument('--queue_param', default=None)
    parser.add_argument('--output_path', default='/results')
    parser.add_argument('--max_people', default=2)
    parser.add_argument('--threshold', default=0.60)
    
    args=parser.parse_args()

    main(args)