#include "nnseq.h"

t_symbol* activation_to_symbol(t_layer *l)
{
  if (l->activation < ACTIVATION_LINEAR || l->activation > ACTIVATION_RELU) {
    post("WARNING: Invalid activation value: %d", l->activation);
    return gensym("INVALID");
  }

  switch(l->activation) {
    case ACTIVATION_SIGMOID:
      return gensym("sigmoid");
    case ACTIVATION_TANH:
      return gensym("tanh");
    case ACTIVATION_RELU:
      return gensym("relu");
    case ACTIVATION_LINEAR:
      return gensym("linear");
    default:
      // this should never be reached
      return gensym("UNKNOWN"); 
  }
}

void set_layer_activation(t_nnseq *x, t_symbol *s, int argc, t_atom *argv)
{
  if (argc != 2) {
    pd_error(x, "fmodel: set_activation requires layer_index and activation_type");
    return;
  }

  int layer_idx = atom_getint(argv++);
  t_symbol *act_type = atom_getsymbol(argv++);
  post("act_type: %s", act_type->s_name);

  if (layer_idx < 0 || layer_idx >= x->num_layers) {
    pd_error(x, "fmodel: invalid layer index %d", layer_idx);
    return;
  }

  if (act_type == gensym("sigmoid")) {
    x->layers[layer_idx].activation = ACTIVATION_SIGMOID;
  } else if (act_type == gensym("tanh")) {
    x->layers[layer_idx].activation = ACTIVATION_TANH;
  } else if (act_type == gensym("relu")) {
    x->layers[layer_idx].activation = ACTIVATION_RELU;
  } else if (act_type == gensym("none")) {
    x->layers[layer_idx].activation = ACTIVATION_LINEAR;
  } else {
    pd_error(x, "fmodel: unknown activation type '%s'", act_type->s_name);
  }
}

void get_layer_info(t_nnseq *x)
{
  post("Model has %d layers", x->num_layers);
  post("Batch size: %d", x->batch_size);

  for (int i = 0; i < x->num_layers; i++) {
    t_layer *layer = &x->layers[i];

    post("layer: %d", i);
    post("neurons: %d", layer->n);
    post("inputs per neuron: %d", layer->n_prev);
    post("activation: %s", activation_to_symbol(layer)->s_name);

    if (layer->a_cache) {
      post("Layer activations:");
      int activations_size = layer->n * x->batch_size;
      for (int j = 0; j < activations_size; j++) {
        post("a_cache[%d]: %f", j, layer->a_cache[j]);
      }
    } else {
      post("activations cache not allocated");
    }

    if (layer->weights) {
      post("Sample weights (first 3 or fewer):");
      int sample_size = layer->n * layer->n_prev < 3 ? layer->n * layer->n_prev : 3;
      for (int j = 0; j < sample_size; j++) {
        post("weight[%d]: %f", j, layer->weights[j]);
        }
      } else {
      post("weights not allocated");
    }

    if (layer->biases) {
      post("sample biases (first 3 or fewer):");
      int sample_size = layer->n < 3 ? layer->n : 3;
      for (int j = 0; j < sample_size; j++) {
        post("bias[%d]: %f", j, layer->biases[j]);
      }
    } else {
      post("biases not allocated");
    }
  }
}

t_float apply_activation(t_nnseq *x, t_int l, t_float z)
{
  t_layer *layer = &x->layers[l];
  switch(layer->activation) {
    case ACTIVATION_SIGMOID:
      return 1.0 / (1.0 + exp(-z));
    case ACTIVATION_TANH:
      return tanh(z);
    case ACTIVATION_RELU:
      return z > 0 ? z : 0;
    case ACTIVATION_LINEAR:
    default:
      return z;
  }
}

float he_init(int n_prev)
{
  float u1 = (float)rand() / RAND_MAX;
  float u2 = (float)rand() / RAND_MAX;
  float radius = sqrt(-2 * log(u1));
  float theta = 2 * M_PI * u2;
  float standard_normal = radius * cos(theta);

  return standard_normal * sqrt(2.0 / n_prev);
}


void init_layer_weights(t_nnseq *x, t_int l)
{
  t_layer *layer = &x->layers[l];
  int w_size = layer->n * layer->n_prev;
  for (int i = 0; i < w_size; i++) {
    layer->weights[i] = he_init(layer->n_prev);
  }
}

void init_layer_biases(t_nnseq *x, t_int l)
{
  t_layer *layer = &x->layers[l];
  for (int i = 0; i < layer->n; i++) {
    layer->biases[i] = 0.0;
  }
}

void get_layer_activation(t_nnseq *x, t_float idx)
{
  // TODO: l might be out of bounds
  int l = (int)idx;
  t_layer *layer = &x->layers[l];
  t_symbol *act = activation_to_symbol(layer);
  post("activation for layer %d: %s", l, act->s_name);
}
