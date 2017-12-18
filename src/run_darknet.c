#include <darknet.h>

float *run_net
    (
    char *cfgfile,
    char *weightfile,
    float *indata,
    int *outw,
    int *outh
    )
{
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    *outw = net->layers[net->n - 2].out_w;
    *outh = net->layers[net->n - 2].out_h;
    return network_predict(net, indata);
}
