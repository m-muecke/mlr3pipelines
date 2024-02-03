#' @title Principle Component Analysis
#'
#' @usage NULL
#' @name mlr_pipeops_umap
#' @format [`R6Class`] object inheriting from [`PipeOpTaskPreproc`]/[`PipeOp`].
#'
#' @description
#' Extracts principle components from data. Only affects numerical features.
#' See [uwot::umap()] for details.
#'
#' @section Construction:
#' ```
#' PipeOpUMAP$new(id = "umap", param_vals = list())
#' ```
#'
#' * `id` :: `character(1)`\cr
#'   Identifier of resulting object, default `"umap"`.
#' * `param_vals` :: named `list`\cr
#'   List of hyperparameter settings, overwriting the hyperparameter settings that would otherwise be set during construction. Default `list()`.
#'
#' @section Input and Output Channels:
#' Input and output channels are inherited from [`PipeOpTaskPreproc`].
#'
#' The output is the input [`Task`][mlr3::Task] with all affected numeric features replaced by their principal components.
#'
#' @section State:
#' The `$state` is a named `list` with the `$state` elements inherited from [`PipeOpTaskPreproc`], as well as the elements of the class [stats::prcomp],
#' with the exception of the `$x` slot. These are in particular:
#' * `sdev` :: `numeric`\cr
#'   The standard deviations of the principal components.
#' * `rotation` :: `matrix`\cr
#'   The matrix of variable loadings.
#' * `center` :: `numeric` | `logical(1)`\cr
#'   The centering used, or `FALSE`.
#' * `scale` :: `numeric` | `logical(1)`\cr
#'   The scaling used, or `FALSE`.
#'
#' @section Parameters:
#' The parameters are the parameters inherited from [`PipeOpTaskPreproc`], as well as:
#' * `center` :: `logical(1)`\cr
#'   Indicating whether the features should be centered. Default is `TRUE`. See [`prcomp()`][stats::prcomp].
#' * `scale.` :: `logical(1)`\cr
#'   Whether to scale features to unit variance before analysis. Default is `FALSE`, but scaling is advisable. See [`prcomp()`][stats::prcomp].
#' * `rank.` :: `integer(1)`\cr
#'   Maximal number of principal components to be used. Default is `NULL`: use all components. See [`prcomp()`][stats::prcomp].
#'
#' @section Internals:
#' Uses the [`umap()`][uwot::umap] function.
#'
#' @section Methods:
#' Only methods inherited from [`PipeOpTaskPreproc`]/[`PipeOp`].
#'
#' @examples
#' library("mlr3")
#'
#' task = tsk("iris")
#' pop = po("umap")
#'
#' task$data()
#' pop$train(list(task))[[1]]$data()
#'
#' pop$state
#' @family PipeOps
#' @template seealso_pipeopslist
#' @include PipeOpTaskPreproc.R
#' @export
PipeOpUMAP = R6Class("PipeOpUMAP",
  inherit = PipeOpTaskPreproc,
  public = list(
    initialize = function(id = "umap", param_vals = list()) {
      ps = ps(
        n_neighbors = p_int(2L, 100L, default = 15L, tags = c("train", "umap")),
        n_components = p_int(1L, 100L, default = 2L, tags = c("train", "umap")),
        metric = p_fct(c("euclidean", "cosine", "manhattan", "hamming", "correlation", "categorical"), default = "euclidean", tags = c("train", "umap")),
        n_epochs = p_int(1L, default = NULL, special_vals = list(NULL), tags = c("train", "umap")),
        learning_rate = p_dbl(0, default = 1, tags = c("train", "umap")),
        scale = p_lgl(default = FALSE, special_vals = list("none", "Z", "maxabs", "range", "colrange", NULL), tags = c("train", "umap")),
        # TODO: also allows A matrix of initial coordinates.
        init = p_fct(c("spectral", "normlaplacian", "random", "lvrandom", "laplacian", "pca", "spca", "agspectral"), default = "spectral", tags = c("train", "umap")),
        init_sdev = p_dbl(0, default = NULL, special_vals = list(NULL), tags = c("train", "umap")),
        spread = p_dbl(0, default = 1, tags = c("train", "umap")),
        min_dist = p_dbl(0, default = 0.01, tags = c("train", "umap")),
        set_op_mix_ratio = p_dbl(0, 1, default = 1, tags = c("train", "umap")),
        local_connectivity = p_int(1L, 100L, default = 1L, tags = c("train", "umap")),
        bandwidth = p_dbl(0, default = 1, tags = c("train", "umap")),
        repulsion_strength = p_dbl(0, default = 1, tags = c("train", "umap")),
        negative_sample_rate = p_int(1L, 100L, default = 5L, tags = c("train", "umap")),
        # a
        # b
        nn_method = p_fct(c("fnn", "annoy"), default = "annoy", tags = c("train", "umap")),
        n_trees = p_int(10L, 100L, default = 50L, tags = c("train", "umap")),
        search_k = p_int(1L, 100L, default = 15L, tags = c("train", "umap")),
        approx_pow = p_lgl(default = FALSE, tags = c("train", "umap")),
        target_weight = p_dbl(0, 1, default = 0.5, tags = c("train", "umap")),
        target_n_neighbors = p_int(1L, 100L, default = 15L, tags = c("train", "umap")),
        target_metric = p_fct(c("euclidean", "cosine", "manhattan", "hamming", "correlation", "categorical"), default = "euclidean", tags = c("train", "umap")),
        pca = p_lgl(default = NULL, special_vals = list(NULL), tags = c("train", "umap")),
        pca_center = p_lgl(default = TRUE, tags = c("train", "umap")),
        pca_rand = p_lgl(default = TRUE, tags = c("train", "umap")),
        pca_method = p_fct(c("irlba", "rsvd", "bigstatsr", "svd", "auto"), default = NULL, special_vals = list(NULL), tags = c("train", "umap")),
        fast_sgd = p_lgl(default = FALSE, tags = c("train", "umap")),
        n_threads = p_int(1L, default = NULL, special_vals = list(NULL), tags = c("train", "umap")),
        n_sgd_threads = p_int(0L, default = 0L, tags = c("train", "umap")),
        grain_size = p_int(1L, default = 1L, tags = c("train", "umap")),
        verbose = p_lgl(default = TRUE, tags = c("train", "umap")),
        batch = p_lgl(default = FALSE, tags = c("train", "umap")),
        binary_edge_weights = p_lgl(default = FALSE, tags = c("train", "umap")),
        dens_scale = p_dbl(0, 1, default = NULL, special_vals = list(NULL), tags = c("train", "umap")),
        seed = p_int(default = NULL, special_vals = list(NULL), tags = c("train", "umap"))
      )
      super$initialize(id, param_set = ps, param_vals = param_vals, feature_types = c("numeric", "integer"))
      # param deps
      # ps$add_dep("tweedie_variance_power", "objective", CondEqual$new("reg:tweedie"))
      # # custom defaults
      # param_set$set_values(
      #   # drop = TRUE,
      # )
    }
  ),
  private = list(

    .train_dt = function(dt, levels, target) {
      params = insert_named(self$param_set$get_values(tags = "umap"), list(ret_model = TRUE))
      umap = invoke(uwot::umap, dt, .args = params)
      self$state = umap
      self$state$embedding = NULL
      umap$embedding
    },

    .predict_dt = function(dt, levels) {
      uwot::umap_transform(dt, self$state)
    }
  )
)

mlr_pipeops$add("umap", PipeOpUMAP)
