#include "nnseq.h"

static t_class *nnseq_class = NULL;


static void nnseq_free(t_nnseq *x)
{
  if (x->x_input != NULL) {
    freebytes(x->x_input, x->layer_dims[0] * x->batch_size * sizeof(t_float));
  }

  if (x->y_labels != NULL) {
    freebytes(x->y_labels, x->layer_dims[x->num_layers] * x->batch_size *
              sizeof(t_float));
  }

  if (x->layers != NULL) {
    for (int i = 0; i < x->num_layers; i++) {
      if (x->layers[i].weights != NULL) {
        freebytes(x->layers[i].weights, x->layers[i].n * x->layers[i].n_prev *
                sizeof(t_float));
      }

      if (x->layers[i].biases != NULL) {
        freebytes(x->layers[i].biases, x->layers[i].n * sizeof(t_float));
      }

      if (x->layers[i].z_cache != NULL) {
        freebytes(x->layers[i].z_cache, x->layers[i].n * x->batch_size * sizeof(t_float));
      }

      if (x->layers[i].a_cache != NULL) {
        freebytes(x->layers[i].a_cache, x->layers[i].n * x->batch_size * sizeof(t_float));
      }
    }
    freebytes(x->layers, x->num_layers * sizeof(t_layer));
  }

  if (x->layer_dims != NULL) {
    freebytes(x->layer_dims, (x->num_layers + 1) * sizeof(int));
  }

  if (x->output_outlet != NULL) {
    outlet_free(x->output_outlet);
  }
}


static int init_layers(t_nnseq *x)
{
  x->layers = (t_layer *)getbytes(sizeof(t_layer) * x->num_layers);
  if (x->layers == NULL) {
    pd_error(x, "nnseq: failed to allocate memory for model layers");
    return 0;
  }

  srand(time(NULL)); // random seed (only call once, maybe not here)

  for (int l = 0; l < x->num_layers; l++) {
    t_layer *layer = &x->layers[l];
    layer->n = x->layer_dims[l+1];
    layer->n_prev = x->layer_dims[l];

    if (l < x->num_layers - 1) {
      layer->activation = ACTIVATION_RELU; // default to ReLU
    } else {
      layer->activation = ACTIVATION_LINEAR;
    }

    // allocate memory for layer params and caches
    layer->weights = (t_float *)getbytes(sizeof(t_float) * layer->n * layer->n_prev);
    if (layer->weights == NULL) {
      pd_error(x, "nnseq: failed to allocate memory for layer weights");
      return 0;
    }
    layer->biases = (t_float *)getbytes(sizeof(t_float) * layer->n);
    if (layer->biases == NULL) {
      pd_error(x, "nnseq: failed to allocate memory for layer biases");
      return 0;
    }
    layer->z_cache = (t_float *)getbytes(sizeof(t_float) * layer->n * x->batch_size);
    if (layer->z_cache == NULL) {
      pd_error(x, "nnseq: failed to allocate memory for layer z_cache");
      return 0;
    }
    layer->a_cache = (t_float *)getbytes(sizeof(t_float) * layer->n * x->batch_size);
    if (layer->a_cache == NULL) {
      pd_error(x, "nnseq: failed to allocate memory for a_cache");
      return 0;
    }

    init_layer_weights(x, l);
    init_layer_biases(x, l);
  }

  return 1;
}

static void nnseq_bang(t_nnseq *x)
{
  post("in the nnseq bang method");
}

static void *nnseq_new(t_symbol *s, int argc, t_atom *argv)
{
  t_nnseq *x = (t_nnseq *)pd_new(nnseq_class);

  // args are batch_size, [layer_dims] (for now)
  if (argc < 3) {
    pd_error(x, "nnseq: batch_size and input/output dimensions must be set");
    nnseq_free(x);
    return NULL;
  }

  x->layers = NULL;
  x->layer_dims = NULL;
  x->x_input = NULL;
  x->y_labels = NULL;

  // first arg is batch_size
  x->batch_size = atom_getfloat(argv++);
  
  // the number of network layers excludes the input layer
  x->num_layers = argc - 2;

  // layer_dims includes the input layer
  x->layer_dims = (int *)getbytes((x->num_layers + 1) * sizeof(int));
  if (x->layer_dims == NULL) {
    pd_error(x, "nnseq: failed to allocate memory for layer_dims");
    nnseq_free(x);
    return NULL;
  }

  // store all dimensions (input features at index 0, then hidden layers, then
  // output layer). Note the upper bound uses <=.
  for (int i = 0; i <= x->num_layers; i++) {
    x->layer_dims[i] = atom_getfloat(argv++);
  }

  if (!init_layers(x)) {
    nnseq_free(x);
    return NULL;
  }

  x->output_outlet = outlet_new(&x->x_obj, &s_list);

  return (void *)x;
}

void nnseq_setup(void)
{
  nnseq_class = class_new(gensym("nnseq"),
                           (t_newmethod)nnseq_new,
                           (t_method)nnseq_free,
                           sizeof(t_nnseq),
                           CLASS_DEFAULT,
                           A_GIMME,
                           0);

  class_addbang(nnseq_class, nnseq_bang);
}
