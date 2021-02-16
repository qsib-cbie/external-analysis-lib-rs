#![feature(default_alloc_error_handler)]
#![feature(lang_items)]
#![feature(integer_atomics)]
#![feature(drain_filter)]
#![feature(test)]
#![allow(soft_unstable)]
#![allow(dead_code)]

extern crate alloc;
use alloc::vec::Vec;
use cstr_core::{c_char, CString};
use lazy_static::lazy_static;
use spin::Mutex;

use std::{
    collections::HashMap,
    ffi::CStr,
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc, RwLock,
    },
};

mod version;

const SKIN_HYDRATION_SENSOR_V2: &str = "Skin Hydration V2";

lazy_static! {
    static ref ERRORS: RwLock<Vec<CString>> = RwLock::new(Vec::new());
    static ref DATASETS: RwLock<HashMap<u32, Arc<Mutex<RwDataset>>>> = RwLock::new(HashMap::new());
    static ref DATASET_ID: AtomicU32 = AtomicU32::new(1);
}

#[repr(C)]
pub struct AnalysisLibPeripheral {
    pub project_mode: *const c_char,
    pub firmware_version: *const c_char,
    pub hardware_version: *const c_char,
    pub boot_count: u32,
    pub calibration_factor0: f32,
    pub calibration_factor1: f32,
    pub calibration_factor2: f32,
    pub calibration_factor3: f32,
    pub aux_buf: *mut u8,
    pub aux_buf_len: u32,
}

#[repr(C)]
pub struct AnalysisLibDataset {
    pub dataset_id: u32,
    pub timestamps: *const f64,
    pub num_channels: u32,
    pub channels: *const *const f64,
    pub num_samples: u32,
}

impl Default for AnalysisLibDataset {
    fn default() -> Self {
        AnalysisLibDataset {
            dataset_id: 0,
            timestamps: core::ptr::null(),
            num_channels: 0,
            channels: core::ptr::null(),
            num_samples: 0,
        }
    }
}

struct AnalysisLibPeripheralInternal {
    pub project_mode: String,
    pub firmware_version: String,
    pub hardware_version: String,
    pub boot_count: u32,
    pub calibration_factor0: f32,
    pub calibration_factor1: f32,
    pub calibration_factor2: f32,
    pub calibration_factor3: f32,
    pub aux_buf: *mut u8,
    pub aux_buf_len: u32,
}

struct AnalysisLibInternalDataset {
    timestamps: Vec<f64>,
    channel_ptrs: Vec<*const f64>,
    channels: Vec<Vec<f64>>,
}

struct RwDataset {
    id: u32,
    available: bool,
    dataset: AnalysisLibInternalDataset,
}
unsafe impl Send for RwDataset {}

impl AnalysisLibPeripheralInternal {
    pub fn from(external: *const AnalysisLibPeripheral) -> AnalysisLibPeripheralInternal {
        unsafe {
            AnalysisLibPeripheralInternal {
                project_mode: CStr::from_ptr((*external).project_mode as *const i8)
                    .to_string_lossy()
                    .into_owned(),
                firmware_version: CStr::from_ptr((*external).firmware_version as *const i8)
                    .to_string_lossy()
                    .into_owned(),
                hardware_version: CStr::from_ptr((*external).hardware_version as *const i8)
                    .to_string_lossy()
                    .into_owned(),
                boot_count: (*external).boot_count,
                calibration_factor0: (*external).calibration_factor0,
                calibration_factor1: (*external).calibration_factor1,
                calibration_factor2: (*external).calibration_factor2,
                calibration_factor3: (*external).calibration_factor3,
                aux_buf: (*external).aux_buf,
                aux_buf_len: (*external).aux_buf_len,
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn analysis_lib_init() {
    // nop
    env_logger::init();
    log::debug!(
        "analysis_lib_init: RUST_LOG={:?}",
        std::env::var("RUST_LOG")
    );
}

#[no_mangle]
pub extern "C" fn analysis_lib_version_get() -> *mut c_char {
    log::debug!("analysis_lib_version_get");
    CString::new(format!("{}", version::get_BUILD_VERSION()))
        .unwrap()
        .into_raw()
}

#[no_mangle]
pub extern "C" fn analysis_lib_version_release(cstr: *mut c_char) {
    log::debug!("analysis_lib_version_release");
    unsafe {
        if !cstr.is_null() {
            CString::from_raw(cstr);
        }
    }
}

#[no_mangle]
pub extern "C" fn analysis_lib_errors_pop() -> *mut c_char {
    log::debug!("analysis_lib_errors_pop");
    let mut error_guard = ERRORS.write().unwrap();
    match (*error_guard).pop() {
        Some(message) => {
            return message.into_raw();
        }
        None => {
            return core::ptr::null_mut();
        }
    }
}

#[no_mangle]
pub extern "C" fn analysis_lib_error_release(cstr: *mut c_char) {
    log::debug!("analysis_lib_error_release");
    unsafe {
        if !cstr.is_null() {
            CString::from_raw(cstr);
        }
    }
}

#[no_mangle]
pub extern "C" fn analysis_lib_dataset_release(dataset: *mut AnalysisLibDataset) -> bool {
    log::debug!("analysis_lib_dataset_release");

    // Get write access to the structure with metadata on existing datasets
    let mut datasets_guard = DATASETS.write().unwrap();

    // Find the dataset and make it available, returning false if nothing was actually released (made available)
    let dataset_id = unsafe { (*dataset).dataset_id };
    match (*datasets_guard).get_mut(&dataset_id) {
        Some(dataset) => {
            let mut g = dataset.lock();
            if g.available {
                false
            } else {
                g.available = true;
                true
            }
        }
        None => false,
    }
}

#[no_mangle]
pub extern "C" fn analysis_lib_analyze(
    params: *const AnalysisLibPeripheral,
    datasets: *const AnalysisLibDataset,
    num_datasets: u32,
    results: *mut AnalysisLibDataset,
    num_results: *mut u32,
) -> bool {
    log::debug!("analysis_lib_analyze");

    let params = AnalysisLibPeripheralInternal::from(params);
    match params.project_mode.as_str() {
        SKIN_HYDRATION_SENSOR_V2 => {
            analyze_skin_hydration_sensor_v2(&params, datasets, num_datasets, results, num_results)
        }
        _ => {
            log::warn!(
                "No matching project analysis implementation for {:?}",
                params.project_mode
            );
            false
        }
    }
}

fn analyze_skin_hydration_sensor_v2(
    _params: &AnalysisLibPeripheralInternal,
    datasets: *const AnalysisLibDataset,
    num_datasets: u32,
    results: *mut AnalysisLibDataset,
    num_results: *mut u32,
) -> bool {
    log::debug!("analyze_skin_hydration_sensor_v2");

    // Take the most recent dataset
    if num_datasets < 1 {
        let mut error_guard = ERRORS.write().unwrap();
        let errors = &mut *error_guard;
        if errors.len() < 16 {
            errors.push(CString::new("Invalid number of datasets to analyze").unwrap());
        }
        return false;
    }

    // Dataset Under Inspection
    let _dui = unsafe { &(*datasets.offset(num_datasets as isize - 1)) };

    // Pick a result
    let hydration_percentage = 42.0;
    log::trace!(
        "analyze_skin_hydration_sensor_v2::hydration_percentage = {}",
        hydration_percentage
    );

    // Acquire a dataset to use to return results
    let internal_results = match find_or_create_dataset() {
        Ok(r) => r,
        Err(msg) => {
            let mut error_guard = ERRORS.write().unwrap();
            let errors = &mut *error_guard;
            if errors.len() < 16 {
                errors.push(CString::new(msg).unwrap());
            }
            return false;
        }
    };

    // Assign result at the 30 second mark
    let mut guard = internal_results.lock();
    let rwd = &mut (*guard);
    let d = &mut rwd.dataset;
    d.timestamps.resize(1, 0f64);
    d.channels.resize(1, vec![]);
    d.channel_ptrs.resize(1, core::ptr::null_mut());

    d.timestamps[0] = 30.0f64;
    d.channels[0] = vec![hydration_percentage];
    d.channel_ptrs[0] = d.channels[0].as_ptr() as *const f64;

    let r = unsafe { &mut (*results) };
    r.dataset_id = rwd.id;
    r.timestamps = d.timestamps.as_ptr() as *const f64;
    r.num_samples = 1;
    r.channels = d.channel_ptrs.as_ptr() as *const *const f64;
    r.num_channels = 1;

    unsafe {
        *num_results = 1;
    }

    true
}

fn find_or_create_dataset() -> Result<Arc<Mutex<RwDataset>>, &'static str> {
    log::debug!("find_or_create_dataset");

    // Get write access to the structure with metadata on existing datasets
    let mut datasets_guard = DATASETS
        .write()
        .map_or_else(|_| Err("Failed to acquire access to datasets"), |g| Ok(g))?;

    // Find the first dataset that no one has returned
    for dataset in (*datasets_guard).iter_mut() {
        let mut g = dataset.1.lock();
        if g.available {
            g.available = false;
            return Ok(dataset.1.clone());
        }
    }

    // Fail if there are too many existing datasets
    if (*datasets_guard).len() > 32 {
        return Err(
            "Datasets are not being released. Failing to acquire a dataset to export results.",
        );
    }

    // Allocate a new dataset
    let dataset_id = DATASET_ID.fetch_add(1, Ordering::SeqCst);
    let dataset = Arc::new(Mutex::new(RwDataset {
        id: dataset_id,
        available: false,
        dataset: AnalysisLibInternalDataset {
            timestamps: vec![],
            channel_ptrs: vec![],
            channels: vec![],
        },
    }));

    // Ignore return since we will never re-insert on same key
    let _ = (*datasets_guard).insert(dataset_id, dataset.clone());

    Ok(dataset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    use csv;
    use dotenv;
    use serde::Deserialize;

    lazy_static! {
        static ref TEST_ENV_LOGGER: AtomicU32 = {
            println!("init bootstrap");
            dotenv::dotenv().ok();
            env_logger::init();
            AtomicU32::new(0)
        };
    }

    fn init() {
        println!("init {}", TEST_ENV_LOGGER.fetch_add(1, Ordering::SeqCst));
    }

    #[test]
    fn it_works() {
        init();
        assert_eq!(2 + 2, 4);
    }

    #[allow(non_snake_case)]
    #[derive(Debug, Deserialize)]
    struct SHV2Record {
        TimestampSinceCaptureStart: f64,
        Channel0: f64,
        Channel1: f64,
        Channel2: f64,
        Channel3: f64,
    }

    #[test]
    fn analyzes_dummy_skin_hydration_v2() {
        init();

        // Read in the data from a csv file
        let mut dataset = AnalysisLibInternalDataset {
            timestamps: vec![],
            channel_ptrs: vec![],
            channels: vec![vec![], vec![], vec![]],
        };
        let mut rdr =
            csv::Reader::from_path("./tests/samples/skin-hydration-v2/data_1.csv").unwrap();
        let dataset = rdr.deserialize().fold(&mut dataset, |acc, r| {
            let r: SHV2Record = r.unwrap();
            acc.timestamps.push(r.TimestampSinceCaptureStart);
            acc.channels[0].push(r.Channel0);
            acc.channels[1].push(r.Channel1);
            acc.channels[2].push(r.Channel2);
            acc
        });
        dataset.channel_ptrs = vec![
            dataset.channels[0].as_ptr(),
            dataset.channels[1].as_ptr(),
            dataset.channels[2].as_ptr(),
        ];

        // Prep the params for the call
        let project_mode = CString::new(SKIN_HYDRATION_SENSOR_V2).unwrap();
        let project_mode = project_mode.as_c_str();
        let firmware_version = CString::new("v0.0.0").unwrap();
        let firmware_version = firmware_version.as_c_str();
        let hardware_version = CString::new("v0.0.0").unwrap();
        let hardware_version = hardware_version.as_c_str();
        let params = AnalysisLibPeripheral {
            project_mode: project_mode.as_ptr(),
            firmware_version: firmware_version.as_ptr(),
            hardware_version: hardware_version.as_ptr(),
            boot_count: 42,
            calibration_factor0: 2f32,
            calibration_factor1: 3f32,
            calibration_factor2: 5f32,
            calibration_factor3: 7f32,
            aux_buf: core::ptr::null_mut(),
            aux_buf_len: 0u32,
        };
        let dataset = AnalysisLibDataset {
            dataset_id: 0,
            timestamps: dataset.timestamps.as_ptr(),
            num_channels: 3,
            channels: dataset.channel_ptrs.as_ptr(),
            num_samples: dataset.timestamps.len() as u32,
        };
        let num_datasets: u32 = 1;

        // Results will get shoved here
        let results: Vec<AnalysisLibDataset> = vec![Default::default()];
        let mut num_results: u32 = 1;

        // Run the analysis and verify the expected results
        let success = analysis_lib_analyze(
            &params as *const AnalysisLibPeripheral,
            &dataset as *const AnalysisLibDataset,
            num_datasets,
            results.as_ptr() as *mut AnalysisLibDataset,
            &mut num_results,
        );

        let error = analysis_lib_errors_pop();
        if !error.is_null() {
            let msg = unsafe {
                CStr::from_ptr(error as *const i8)
                    .to_string_lossy()
                    .to_owned()
            };
            println!("Failed with message: {:?}", msg);
            assert!(false);
        }
        assert!(success);
        assert_eq!(num_results, 1);
        assert_eq!(results[0].num_samples, 1);
        assert_eq!(results[0].num_channels, 1);

        unsafe {
            assert_approx_eq!(results[0].timestamps.read(), 30f64, 1e-6f64);
            assert_approx_eq!(results[0].channels.read().read(), 42f64, 1e-5f64);
        }
    }
}
