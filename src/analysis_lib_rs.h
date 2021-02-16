#include <stdint.h>
#include <stdbool.h>

struct analysis_lib_peripheral_t
{
    const char *project_mode;
    const char *firmware_version;
    const char *hardware_version;
    uint32_t boot_count;
    float calibration_factor0;
    float calibration_factor1;
    float calibration_factor2;
    float calibration_factor3;
    uint8_t *aux_buf;
    uint32_t aux_buf_len;
};

struct analysis_lib_dataset_t
{
    uint32_t dataset_id; // meaningless for datasets not returned by params
    const double *timestamps;
    uint32_t num_samples;
    const double *const *channels;
    uint32_t num_channels;
};

/*!
 * This must be called before using the library.
 * It may be called several times and is thread-safe.
 */
void analysis_lib_init();

/*!
 * Get a human readable string VERSION for this library
 *
 * A library owned C String w/ nul-terminator is returned
 * with the expectation of the user calling the drop function.
 */
const char *analysis_lib_version_get();
void analysis_lib_version_release(const char *free_me_please);

/*!
 * Get a human readable error message for this library if any have been generated
 *
 * A library owned C String w/ nul-terminator is returned
 * with the expectation of the user calling the drop function.
 */
const char *analysis_lib_errors_pop();
void analysis_lib_error_release(const char *free_me_please);

/*!
 * Allows re-use of buffers assoicated with the provided id.
 * 
 * If buffers are not returned, they are re-allocated every time and never free'd.
 * 
 * @param[in] dataset The dataset to release from a preceding analysis_lib_analyze.
 * 
 * @return true if the buffer was checked out (meaning you did return a valid buffer id)
 */
bool analysis_lib_dataset_release(struct analysis_lib_dataset_t *dataset);

/*!
 * This method is thread-safe.
 *
 * Perform the analysis on the data provided
 * 
 * @param[in] params   A parameters structure instance describing the peripheral that produced the dataset.
 * @param[in] datasets A sequence of time series data sets to analyze in buffers.
 *                       This memory *is not* managed by the analysis library.
 * @param[in] num_datasets The number of datasets provided.
 * @param[out] results A sequence of time series data sets deduced from analyzation of the provided datasets.
 *                       This memory *is* managed by the anaylysis library with the corresponding buffer_id.
 * @param[out] num_results The number of datasets in results.
 * 
 * @return success if buffer_id is non-zero, it should be returned at some point
 */
bool analysis_lib_analyze(
    const struct analysis_lib_peripheral_t *params,
    const struct analysis_lib_dataset_t *datasets, uint32_t num_datasets,
    struct analysis_lib_dataset_t *results, uint32_t *num_results);