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

typedef enum {
  OPTIMIZATION_GD,
  OPTIMIZATION_L2,
  OPTIMIZATION_MOMENTUM,
  OPTIMIZATION_RMSPROP,
  OPTIMIZATION_ADAM
} t_optimizer;

typedef struct _output_config {
  int *layer_indices; // array of layer indices to output
  int num_layers_to_output;
  int output_activations; // boolean: output activations?
  int output_gradients; // boolean: output gradients? (I think it's going to
  // need to be more specific)
} t_output_config;

typedef struct _layer {
  int n;
  int n_prev;
  t_activation_type activation;

  t_float *weights;
  t_float *dw;
  t_float *v_dw;
  t_float *s_dw;
  t_float *biases;
  t_float *db;
  t_float *v_db;
  t_float *s_db;
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
  t_float beta;
  t_float beta_rmsprop;
  t_optimizer optimizer;

  int iterator; // Current step in the sequence

  t_layer *layers;

  t_float *x_input;
  t_float *y_labels;

  t_inlet *input_inlet;
  t_outlet *output_outlet; // maybe rename?
  t_outlet *seq_outlet;
  t_output_config output_config;
} t_nnseq;

t_symbol* activation_to_symbol(t_layer *l);
void set_layer_activation(t_nnseq *x, t_symbol *s, int argc, t_atom *argv);
void set_optimization_method(t_nnseq *x, t_symbol *optimizer);
void model_info(t_nnseq *x);
t_float apply_activation(t_nnseq *x, t_int l, t_float z);
float he_init(int n_prev);
void init_layer_weights(t_nnseq *x, t_int l);
void init_layer_biases(t_nnseq *x, t_int l);
void get_layer_activation(t_nnseq *x, t_float idx);
