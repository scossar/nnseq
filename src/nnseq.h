#include "m_pd.h"
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef enum {
  ACTIVATION_LINEAR,
  ACTIVATION_SIGMOID,
  ACTIVATION_TANH,
  ACTIVATION_RELU
} t_activation_type;

typedef struct _layer {
  int n;
  int n_prev;
  t_activation_type activation;

  t_float *weights;
  t_float *dw;
  t_float *biases;
  t_float *db;
  t_float *z_cache;
  t_float *dz;
  t_float *a_cache;
  t_float *da;
} t_layer;

typedef struct _nnseq {
  t_object x_obj;

  int num_layers;
  int *layer_dims;
  int batch_size;
  t_float alpha;
  t_float leak;
  t_float lambda;

  int iterator; // Current step in the sequence

  t_layer *layers;

  t_float *x_input;
  t_float *y_labels;

  t_inlet *input_inlet;
  t_outlet **layer_outlets; // array of outlets for layer activations
  int num_outlets; // number of layer outlets
} t_nnseq;

t_symbol* activation_to_symbol(t_layer *l);
void set_layer_activation(t_nnseq *x, t_symbol *s, int argc, t_atom *argv);
void model_info(t_nnseq *x);
t_float apply_activation(t_nnseq *x, t_int l, t_float z);
float he_init(int n_prev);
void init_layer_weights(t_nnseq *x, t_int l);
void init_layer_biases(t_nnseq *x, t_int l);
void get_layer_activation(t_nnseq *x, t_float idx);
