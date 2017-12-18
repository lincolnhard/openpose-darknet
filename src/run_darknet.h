#ifndef RUN_DARKNET_H
#define RUN_DARKNET_H

extern "C"
{

float *run_net
    (
    char *cfgfile,
    char *weightfile,
    float *indata,
    int *outw,
    int *outh
    );


}
#endif // RUN_DARKNET_H
