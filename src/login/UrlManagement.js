//数据集上传URL
// export const upload_url = "http://127.0.0.1:5000/upload/"
export const dataset_upload_url = 'http://your_port:8000/app/v1/dataset/upload_dataset';

//模型上传URL
// export const model_upload_url = "http://your_port:8000/app/v1/model/upload_premodel"
export const model_upload_url = 'http://your_port:81/api/v1/model/upload_model';
// export const model_upload_url = "/backend/api/v1/model/upload_model"

//数据集列表URL
// export const dataset_list_url = 'http://your_port:81/api/v1/dataset/all_datasets_information'
export const dataset_list_url = '/backend/api/v1/dataset/all_datasets_information';

//预训练模型列表URL
// export const model_list_url = 'http://your_port:81/api/v1/model/all_models_information'
export const model_list_url = '/backend/api/v1/model/all_models_information';

// export const model_list_url = 'http://your_port:81/api/v1/model/model_file_list'

//数据集详细信息URL
export const dataset_url = 'http://your_port:8000/app/v1/dataset/single_dataset_information';
//相关模型URL
// export const relative_url = "http://your_port:81/api/v1/dataset/get_dataset_related_models"
export const relative_url = '/backend/api/v1/dataset/get_dataset_related_models';
//模型详细信息URL
// export const model_url = "http://your_port:8000/app/v1/model/single_models_information"
// export const model_url = "http://your_port:81/api/v1/model/model_file_list"
export const model_url = '/backend/api/v1/model/model_file_list';

/*模型构建*/
//算法列表URL
// export const algorithm_list_url = "http://your_port:8000/app/v1/model/all_algorithms_information"
export const algorithm_list_url = '/backend/app/v1/model/all_algorithms_information';
//数据集列表
export const datasetOption_list_url = '/backend/app/v1/dataset/dataset_version_information';
//预训练模型列表
export const modelOption_list_url = '/backend/app/v1/model/model_version_information';

//启动训练URL
export const train_url = 'http://your_port:8000/app/v1/model/train';

//正在训练的算法模型列表
export const training_model_list_url = 'http://your_port:8000/app/v1/model/train_list';

export const server_url = 'http://your_port:30545/DTEregister/user';

export const solar_cells_defect_detection = 'http://your_port:8432/flask/detect/';
export const solar_cells_fore_detection = 'http://your_port:8432/flask/foreget/';
