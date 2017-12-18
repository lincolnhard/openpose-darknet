# openpose-darknet
Openpose implementation using darknet framework

<b>[Openpose] (official repository)</b><p>
https://github.com/CMU-Perceptual-Computing-Lab/openpose

<b>[EasyOpenPose] (simplified, compressed to one source file)</b><p>
https://github.com/dlunion/EasyOpenPose

<b>[Paper] (show net structure, explain PAFs, buttom-up and greedy algorithm)</b><p>
https://arxiv.org/abs/1611.08050

<b>[Darknet]</b><p>
https://github.com/pjreddie/darknet

<b>[Result]</b><p>
![demo](https://user-images.githubusercontent.com/16308037/34094455-333f678c-e408-11e7-9546-f8aeb3df39c2.jpg)

<b>[Benchmark]</b><p> 51.634ms for 200x200x3 net input size running on GTX 1060, which is 3x faster than original caffe implementation

<b>[Weight file] (darknet version openpose.weight)</b><p>
https://drive.google.com/open?id=1BfY0Hx2d2nm3I4JFh0W1cK2aHD1FSGea
  
<b>[Usage]</b><p>
```Bash
./openpose-darknet [image file] [cfg file] [weight file] [net input width] [net input height]
#example
./openpose-darknet person.jpg openpose.cfg openpose.weight 200 200
```

<b>[note]</b><p>
1. Darknet version openpose.cfg and openpose.weight are ported from COCO version 
  pose_deploy_linevec.prototxt and pose_iter_440000.caffemodel.
2. The argument [net input width] [net input height] should be <b>exactly the same </b>as 
  the net width, net height in openpose.cfg.
