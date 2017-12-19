# openpose-darknet
Openpose implementation using darknet framework

<b>[Openpose] (official repository)</b><p>
https://github.com/CMU-Perceptual-Computing-Lab/openpose

<b>[Darknet]</b><p>
https://github.com/pjreddie/darknet

<b>[Result]</b><p>
![demo](https://user-images.githubusercontent.com/16308037/34094455-333f678c-e408-11e7-9546-f8aeb3df39c2.jpg)

<b>[Benchmark]</b><p> 51.634ms for 200x200x3 net input size running on GTX 1060, which is 3x faster than original caffe implementation

<b>[Weight file] (darknet version openpose.weight)</b><p>
https://drive.google.com/open?id=1BfY0Hx2d2nm3I4JFh0W1cK2aHD1FSGea
  
<b>[Usage]</b><p>
```Bash
./openpose-darknet [image file] [cfg file] [weight file]
#example
./openpose-darknet person.jpg openpose.cfg openpose.weight
```

<b>[note]</b><p>
1. Darknet version openpose.cfg and openpose.weight are ported from COCO version 
  pose_deploy_linevec.prototxt and pose_iter_440000.caffemodel.
2. You could change net input width, height in openpose.cfg.
