'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import time
from openvino.inference_engine import IENetwork, IECore
import cv2
import sys
import logging as log

class Face_detection:
    '''
    Class for the Face Detection Model.
    '''
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        
        try:
            self.model=IENetwork(self.model_structure, self.model_weights)
            #self.model=IECore.read_network(model=self.model_structure, weights=self.model_weights)           
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")        

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self, ie):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
     
        # Read IR
        log.info("Loading network files:\n\t{}\n\t{}".format(self.model_structure, self.model_weights))
        self.net = ie.read_network(model=self.model_structure, weights=self.model_weights)

        #Check supported layers
        if "CPU" in self.device:
            supported_layers = ie.query_network(self.net, "CPU")
            not_supported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                log.error("Layers are not supported {}:\n {}".
                      format(self.device, ', '.join(not_supported_layers)))
                log.error("Specify cpu extensions using -l")
                #sys.exit(1)

        # Load IR to the plugin
        log.info("Loading IR to the plugin...")
        self.exec_net = ie.load_network(network=self.net, num_requests=0, device_name=self.device)
        
        self.input_blob=next(iter(self.exec_net.inputs))
        self.output_blob=next(iter(self.exec_net.outputs))
         

    def predict(self, image, draw_flags):
        '''
        Perform inference.
        '''
        log.info("Performing fd inference...")

        feed_dict = self.preprocess_input(image)
        outputs=self.exec_net.start_async(request_id=0, inputs=feed_dict)
        while True:
            status=self.exec_net.requests[0].wait(-1)
            if status==0:
                break
            else: time.sleep(1)
        coords=self.preprocess_output(outputs)
        print(coords)
        if coords:
            head_image=image[coords[0][1]:coords[0][3], coords[0][0]:coords[0][2]]
        elif not coords:
            head_image=[]
        if 'fd' in draw_flags:
            if coords:
                self.draw_outputs(coords, image)
        return coords, image, head_image

    def preprocess_input(self, image):
        '''
		Preprocess input images and return dictionnary of modified images.
		'''
        #log.info("Preprocessing the input images...")
        input_dict={}
        n, c, h, w = self.input_shape
        in_frame = cv2.resize(image, (w, h))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((n, c, h, w))
        input_dict[self.input_name] = in_frame
        return input_dict

    def preprocess_output(self, outputs):
        '''
		Preprocess the output and return coodinates of BBox(s).
		'''
        res = self.exec_net.requests[0].outputs[self.output_blob]
        coords=[]
        for obj in res[0][0]:
            if obj[2] > 0.6: #args.threshold:           
                xmin = int(obj[3] * self.initial_w)
                ymin = int(obj[4] * self.initial_h)
                xmax = int(obj[5] * self.initial_w)
                ymax = int(obj[6] * self.initial_h)
                coords.append((xmin, ymin, xmax, ymax))
        return coords

    def draw_outputs(self, coords, image):
        '''
        Draw Bounding Boxs and texts on images.
        '''
        color = (10, 245, 10)
        for obj in coords:
            cv2.rectangle(image, (obj[0], obj[1]), (obj[2], obj[3]), color, 2)

    def set_initial(self, w, h):
        self.initial_w = w
        self.initial_h = h 
