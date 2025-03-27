#include "nnseq.h"
#include <m_pd.h>

static t_class *nnseq_class = NULL;

// NOTE: multi-channel-output implementation discussed at
// flattened_matrix_multiplication_with_transforms.md (line ~1196)

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

      // weights and dw
      if (x->layers[i].weights != NULL) {
        freebytes(x->layers[i].weights, x->layers[i].n * x->layers[i].n_prev *
                sizeof(t_float));
      }
      if (x->layers[i].dw != NULL) {
        freebytes(x->layers[i].dw, x->layers[i].n * x->layers[i].n_prev *
                sizeof(t_float));
      }

      // biases and db
      if (x->layers[i].biases != NULL) {
        freebytes(x->layers[i].biases, x->layers[i].n * sizeof(t_float));
      }
      if (x->layers[i].db != NULL) {
        freebytes(x->layers[i].db, x->layers[i].n * sizeof(t_float));
      }

      // z and dz
      if (x->layers[i].z_cache != NULL) {
        freebytes(x->layers[i].z_cache, x->layers[i].n * x->batch_size * sizeof(t_float));
      }
      if (x->layers[i].dz != NULL) {
        freebytes(x->layers[i].dz, x->layers[i].n * x->batch_size * sizeof(t_float));
      }

      // a and da
      if (x->layers[i].a_cache != NULL) {
        freebytes(x->layers[i].a_cache, x->layers[i].n * x->batch_size * sizeof(t_float));
      }
      if (x->layers[i].da != NULL) {
        freebytes(x->layers[i].da, x->layers[i].n * x->batch_size * sizeof(t_float));
      }
    }
    freebytes(x->layers, x->num_layers * sizeof(t_layer));
  }

  if (x->layer_dims != NULL) {
    freebytes(x->layer_dims, (x->num_layers + 1) * sizeof(int));
  }

  if (x->info_outlet != NULL) {
    outlet_free(x->info_outlet);
  }

  if (x->layer_outlets != NULL) {
    freebytes(x->layer_outlets, x->num_outlets * sizeof(t_outlet *));
  }
}

// const t_atom *argv makes it explicit that argv can't be modified
static void set_x_input(t_nnseq *x, t_symbol *s, int argc, const t_atom *argv)
{
  int input_dim = x->layer_dims[0];
  int batch_size = x->batch_size;
  int expected_size = input_dim * batch_size;

  if (expected_size != argc) {
    pd_error(x, "nnseq: wrong number of input elements. expected: %d, got: %d",
             expected_size, argc);
    return;
  }

  // free any existing allocation
  // NOTE: possibly hold off on this until
  // the new values have been validated.
  if (x->x_input != NULL) {
    freebytes(x->x_input, expected_size * sizeof(t_float));
  }

  // allocate memory
  x->x_input = (t_float *)getbytes(expected_size * sizeof(t_float));
  if (x->x_input == NULL) {
    pd_error(x, "nnseq: failed to allocate memory for x_input");
    return;
  }

  // NOTE: on error, any newly allocated memory is freed and x_input is set to
  // NULL. Consider writing to a temporary buffer and preserving existing values
  // until after validation.
  for (int i = 0; i < argc; i++) {
    if (argv[i].a_type != A_FLOAT) {
      pd_error(x, "nnseq: non-numeric input value at index %d", i);
      freebytes(x->x_input, expected_size * sizeof(t_float));
      x->x_input = NULL;
      return;
    }
    t_float value = atom_getfloat(&argv[i]);
    if (isinf(value)) {
      pd_error(x, "nnseq: infinite value at index %d", i);
      freebytes(x->x_input, expected_size * sizeof(t_float));
      x->x_input = NULL;
      return;
    }

    x->x_input[i] = value;
  }
}

static void set_y_labels(t_nnseq *x, t_symbol *s, int argc, const t_atom *argv)
{
  // x->num_layers is one fewer than the number of layer_dims
  int output_dim = x->layer_dims[x->num_layers];
  int batch_size = x->batch_size;
  int expected_size = output_dim * batch_size;

  if (expected_size != argc) {
    pd_error(x, "nnseq: wrong number of Y elements. Expected: %d, got %d",
             expected_size, argc);
    return;
  }
  
  // free any existing allocation
  if (x->y_labels != NULL) {
    freebytes(x->y_labels, expected_size * sizeof(t_float));
  }

  x->y_labels = (t_float *)getbytes(expected_size * sizeof(t_float));
  if (x->y_labels == NULL) {
    pd_error(x, "nnseq: failed to allocate memory for y_labels");
    return;
  }

  for (int i = 0; i < expected_size; i++) {
    if (argv[i].a_type != A_FLOAT) {
      pd_error(x, "nnseq: non-numerical Y value at index %d", i);
      freebytes(x->x_input, expected_size * sizeof(t_float));
      x->x_input = NULL;
      return;
    }
    t_float value = atom_getfloat(&argv[i]);

    if (isinf(value)) {
      pd_error(x, "nnseq: infinite Y value at index %d", i);
      freebytes(x->y_labels, expected_size * sizeof(t_float));
      x->y_labels = NULL;
      return;
    }

    x->y_labels[i] = value;
  }
}

static void get_x_input(t_nnseq *x)
{
  int x_size = x->layer_dims[0] * x->batch_size;

  t_atom *out_atoms = (t_atom *)getbytes(x_size * sizeof(t_atom));
  if (out_atoms == NULL) {
    pd_error(x, "nnseq: failed to allocate memory for x_input buffer");
    return;
  }

  for (int i = 0; i < x_size; i++) {
    SETFLOAT(&out_atoms[i], x->x_input[i]);
  }
  
  outlet_list(x->info_outlet, gensym("list"), x_size, out_atoms);
  freebytes(out_atoms, x_size * sizeof(t_atom));
}


static void get_y_labels(t_nnseq *x)
{
  int y_label_size = x->layer_dims[x->num_layers] * x->batch_size;

  t_atom *out_atoms = (t_atom *)getbytes(y_label_size * sizeof(t_atom));
  if (out_atoms == NULL) {
    pd_error(x, "nnseq: failed to allocate memory for y_label buffer");
    return;
  }

  for (int i = 0; i < y_label_size; i++) {
    SETFLOAT(&out_atoms[i], x->y_labels[i]);
  }

  outlet_list(x->info_outlet, gensym("list"), y_label_size, out_atoms);
  freebytes(out_atoms, y_label_size * sizeof(t_atom));
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
    layer->dw = (t_float *)getbytes(sizeof(t_float) * layer->n * layer->n_prev);
    if (layer->dw == NULL) {
      pd_error(x, "nnseq: failed to allocate memory for layer dw");
      return 0;
    }
    layer->biases = (t_float *)getbytes(sizeof(t_float) * layer->n);
    if (layer->biases == NULL) {
      pd_error(x, "nnseq: failed to allocate memory for layer biases");
      return 0;
    }
    layer->db = (t_float *)getbytes(sizeof(t_float) * layer->n);
    if (layer->db == NULL) {
      pd_error(x, "nnseq: failed to allocate memory for layer db");
      return 0;
    }
    layer->z_cache = (t_float *)getbytes(sizeof(t_float) * layer->n * x->batch_size);
    if (layer->z_cache == NULL) {
      pd_error(x, "nnseq: failed to allocate memory for layer z_cache");
      return 0;
    }
    layer->dz = (t_float *)getbytes(sizeof(t_float) * layer->n * x->batch_size);
    if (layer->dz == NULL) {
      pd_error(x, "nnseq: failed to allocate memory for layer dz");
      return 0;
    }
    layer->a_cache = (t_float *)getbytes(sizeof(t_float) * layer->n * x->batch_size);
    if (layer->a_cache == NULL) {
      pd_error(x, "nnseq: failed to allocate memory for layer a_cache");
      return 0;
    }
    layer->da = (t_float *)getbytes(sizeof(t_float) * layer->n * x->batch_size);
    if (layer->da == NULL) {
      pd_error(x, "nnseq: failed to allocate memory for layer da");
      return 0;
    }

    // TODO: remove
    post("initializing layer %d, n: %d, n_prev: %d", l, layer->n, layer->n_prev);

    init_layer_weights(x, l);
    init_layer_biases(x, l);
  }

  return 1;
}

static void layer_forward(t_nnseq *x, t_int l, t_float *input)
{
  t_layer *layer = &x->layers[l];
  int batch_size = x->batch_size;
  int n_neurons = layer->n;
  int n_inputs = layer->n_prev;

  // process one sample at a time to improve cache locality
  // essentially, the batch_size iteration has been moved to the outer loop.
  for (int j = 0; j < batch_size; j++) {
    for (int i = 0; i < n_neurons; i++) {
      t_float z = layer->biases[i];

      for (int k = 0; k < n_inputs; k++) {
        z += layer->weights[i*n_inputs+k] * input[k*batch_size+j];
      }
      int idx = i*batch_size+j;
      layer->z_cache[idx] = z;
      layer->a_cache[idx] = apply_activation(x, l, z);
    }
  }
}


static void model_forward(t_nnseq *x)
{
  // These checks really only need to be run once. See
  // pd_neural_network_external_layer_info.md (line 2187)
  if (x->x_input == NULL) {
    pd_error(x, "nnseq: no input data provided");
    return;
  }

  for (int l = 0; l < x->num_layers; l++) {
    if (x->layers[l].weights == NULL || x->layers[l].biases == NULL) {
      pd_error(x, "nnseq: layer %d not properly initialized", l);
      return;
    }
  }

  for (int l = 0; l < x->num_layers; l++) {
    t_float *input = l == 0 ? x->x_input : x->layers[l-1].a_cache;
    layer_forward(x, l, input);
  }
}

// NOTE: these should be in the utilities file (and not be static)
static t_float activation_derivative(t_activation_type activation,
                                     t_float z,
                                     t_float a,
                                     t_float leak)
{
  switch(activation) {
    case ACTIVATION_SIGMOID:
      return a * (1.0 - a);
    case ACTIVATION_TANH:
      return 1.0 - a * a;
    case ACTIVATION_RELU:
      return z > 0 ? 1.0 : leak; // note: update for leaky relu
    case ACTIVATION_LINEAR:
    default:
      return 1.0;
  }
}

static void da_outer(t_nnseq *x)
{
  t_layer *output_layer = &x->layers[x->num_layers - 1];
  int num_outputs = output_layer->n;
  int output_size = num_outputs * x->batch_size;
  for (int i = 0; i < output_size; i++) {
    output_layer->da[i]  = output_layer->a_cache[i] - x->y_labels[i];
  }
}

static void calculate_output_layer_da(t_nnseq *x)
{
  t_layer *output_layer = &x->layers[x->num_layers - 1];
  int num_outputs = output_layer->n;

  for (int i = 0; i < num_outputs; i++) {
    for (int j = 0; j < x->batch_size; j++) {
      int idx = i * x->batch_size + j;
      // For MSE loss: dL/dA = 2*(A-Y)/m, simplified to (A - Y)
      output_layer->da[idx] = output_layer->a_cache[idx] - x->y_labels[idx];
    }
  }
}

static void calculate_dz(t_nnseq *x, t_layer *layer)
{
  int n_neurons = layer->n;

  for (int i = 0; i < n_neurons; i++) {
    for (int j = 0; j < x->batch_size; j++) {
      int idx = i * x->batch_size + j;
      layer->dz[idx] = layer->da[idx] *
       activation_derivative(layer->activation,
                             layer->z_cache[idx],
                             layer->a_cache[idx],
                             x->leak);
    }
  }
}
static void set_alpha(t_nnseq *x, t_floatarg f)
{
  x->alpha = f;
}

static void set_leak(t_nnseq *x, t_floatarg f)
{
  x->leak = f;
}

static void calculate_dw(t_nnseq *x, int l, t_layer *layer)
{
  int n_neurons = layer->n;
  int n_inputs = layer->n_prev;
  int batch_size = x->batch_size;
  t_float *prev_activations = (l == 0) ? x->x_input : x->layers[l-1].a_cache;
  
  // Initialize dw to zero
  /*for (int i = 0; i < n_neurons * n_inputs; i++) {*/
  /*  layer->dw[i] = 0.0;*/
  /*}*/
  // try this approach
  memset(layer->dw, 0, sizeof(t_float) * n_neurons * n_inputs);
  
  // Calculate gradients: dW = dZ × A_prev.T
  for (int i = 0; i < n_neurons; i++) {
    for (int j = 0; j < n_inputs; j++) {
      for (int k = 0; k < batch_size; k++) {
        // dZ[i,k] × A_prev[j,k]
        layer->dw[i*n_inputs+j] += layer->dz[i*batch_size+k] * 
                                  prev_activations[j*batch_size+k];
      }
      // Average over batch size
      layer->dw[i*n_inputs+j] /= batch_size;
    }
  }
}

/*
db 1/m sum(dZ)
*/
static void calculate_db(t_nnseq *x, t_layer *layer)
{
  int n_neurons = layer->n;

  // initialize db to zero
  memset(layer->db, 0, sizeof(t_float) * n_neurons);

  for (int i = 0; i < n_neurons; i++) {
    for (int j = 0; j < x->batch_size; j++) {
      layer->db[i] += layer->dz[i*x->batch_size+j];
    }
    layer->db[i] /= x->batch_size;
  }
}

// W.T dZ
// W (n, n_prev); dZ (n, batch_size)
static void calculate_da_prev(t_nnseq *x, int l, t_layer *layer)
{
  int n_neurons = layer->n;
  int n_prev = layer->n_prev;
  t_layer *prev_layer = &x->layers[l-1];
  
  // initialize da_prev to zero
  /*for (int i = 0; i < n_prev * x->batch_size; i++) {*/
  /*  prev_layer->da[i] = 0.0;*/
  /*}*/
  memset(layer->da, 0, sizeof(t_float) * n_neurons * x->batch_size);
  
  // da_prev = W^T * dz
  for (int k = 0; k < n_prev; k++) {
    for (int j = 0; j < x->batch_size; j++) {
      for (int i = 0; i < n_neurons; i++) {
        prev_layer->da[k*x->batch_size+j] += layer->weights[i*n_prev+k] * 
                                            layer->dz[i*x->batch_size+j];
      }
    }
  }
}

static void update_parameters(t_nnseq *x)
{
  for (int l = 0; l < x->num_layers; l++) {
    t_layer *layer = &x->layers[l];
    int n_neurons = layer->n;
    int n_inputs = layer->n_prev;

    for (int i = 0; i < n_neurons * n_inputs; i++) {
      layer->weights[i] -= x->alpha * layer->dw[i];
    }

    for (int i = 0; i < n_neurons; i++) {
      layer->biases[i] -= x->alpha * layer->db[i];
    }
  }
}

static void layer_backward(t_nnseq *x, int l)
{
  t_layer *current_layer = &x->layers[l];

  // for output layer, calculate da from loss function
  if (l == x->num_layers - 1) {
    calculate_output_layer_da(x);
  }

  calculate_dz(x, current_layer);
  calculate_dw(x, l, current_layer);
  calculate_db(x, current_layer);

  // calculate da_prev, except for input layer
  if (l > 0) {
    calculate_da_prev(x, l, current_layer);
  }
}

// for Adam and momentum, see flattened_matrix_multiplication_with_transforms.md
// (~ line 756)
static void model_backward(t_nnseq *x)
{
  for (int l = x->num_layers - 1; l >= 0; l--) {
    layer_backward(x, l);
  }
}

static t_float compute_cost(t_nnseq *x)
{
  t_float cost = 0.0;
  t_layer *output_layer = &x->layers[x->num_layers - 1];
  int output_size = output_layer->n;

  for (int i = 0; i < output_size; i++) {
    for (int j = 0; j < x->batch_size; j++) {
      int idx = i * x->batch_size + j;
      t_float diff = output_layer->a_cache[idx] - x->y_labels[idx];
      cost += diff * diff;
    }
  }
  return cost / (2.0 * x->batch_size);
}

static void get_cost(t_nnseq *x)
{
  t_float cost = compute_cost(x);
  outlet_float(x->info_outlet, cost);
}

// NOTE: the info_outlet is causing some confusion here. Probably use an offset
// so that the left-most inlet is reserved for info. Deal with later?
static void nnseq_bang(t_nnseq *x)
{
  model_forward(x);
  model_backward(x);
  update_parameters(x);

  // output activations from (some) layers
  for (int i = 0; i < x->num_outlets; i++) {
    int layer_idx = x->num_layers - 1 - i;
    post("layer index from bang method: %d", layer_idx);
    if (layer_idx < 0) break; // just in case

    t_layer *layer = &x->layers[layer_idx];

    // cycle through all activations in layer
    int activation_idx = x->iterator % (layer->n * x->batch_size);

    // isn't there an unnecessary step here? (probably, but it opens up other
    // options for accessing the values)
    int neuron_idx = activation_idx / x->batch_size;
    int batch_idx = activation_idx % x->batch_size;

    t_float activation = layer->a_cache[neuron_idx * x->batch_size + batch_idx];

    outlet_float(x->layer_outlets[i], activation);
  }
  x->iterator++;
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

  x->alpha = 0.01; // default
  x->leak = 0.01; // default

  // dynamic outlets (first attempt)
  x->num_outlets = x->num_layers > 4 ? 4 : x->num_layers;

  // an array of outlets
  x->layer_outlets = (t_outlet **)getbytes(x->num_outlets * sizeof(t_outlet *));

  // create outlets from right to left
  for (int i = x->num_outlets - 1; i >= 0; i--) {
    x->layer_outlets[i] = outlet_new(&x->x_obj, &s_float);
  }

  // far left outlet (I think)
  x->info_outlet = outlet_new(&x->x_obj, &s_list);

  x->iterator = 0; // tracks the current iteration

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
  class_addmethod(nnseq_class, (t_method)model_info, gensym("info"), 0);
  class_addmethod(nnseq_class, (t_method)set_layer_activation,
                  gensym("set_activation"), A_GIMME, 0);
  class_addmethod(nnseq_class, (t_method)get_layer_activation,
                  gensym("get_activation"), A_FLOAT, 0);
  class_addmethod(nnseq_class, (t_method)set_x_input,
                  gensym("set_x"), A_GIMME, 0);
  class_addmethod(nnseq_class, (t_method)set_y_labels,
                  gensym("set_y"), A_GIMME, 0);
  class_addmethod(nnseq_class, (t_method)get_x_input,
                  gensym("get_x"), 0);
  class_addmethod(nnseq_class, (t_method)get_y_labels,
                  gensym("get_y"), 0);
  class_addmethod(nnseq_class, (t_method)set_alpha,
                  gensym("set_alpha"), A_DEFFLOAT, 0);
  class_addmethod(nnseq_class, (t_method)set_leak,
                  gensym("set_leak"), A_DEFFLOAT, 0);
  class_addmethod(nnseq_class, (t_method)get_cost,
                  gensym("cost"), 0);
}
