#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "init.h"

#ifndef _LEARN_H_
#define _LEARN_H_

network_1layer  *init_network(int o_dim, int v_dim, int b_size, int D, int *N, FILE *fp_log, int program_confirmation);


#endif