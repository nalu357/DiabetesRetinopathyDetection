import sys

sys.dont_write_bytecode = True

params = {}
params["batch_size"] = 1
params["img_shape"] = (256, 256, 1)
params["epochs"] = 40
params["learning_rate"] = 0.001
params["continuing_training"] = False
params["loss_function"] = "mse_loss"
params["d_rate"] = 0.5
params["project_dir"] = "C:\Users\Nalu\Documents\ICB\DiabeticRetinopathyDetection"
params["data_dir"] = params["project_dir"] + "data/"
params["save_predictions_dir"] = params["project_dir"] + "test_predictions/"
params["save_path"] =

# generator params
gen_params = {'dim': (params["img_shape"][0], params["img_shape"][1]),
              'batch_size': params["batch_size"],
              'n_channels': 1,
              'shuffle': True,
              'image_path': params["data_dir"] + "train",
              'label_path': params["data_dir"] + "trainLabels_1.csv"}

multi_input_gen_params = {'small_dim': (params["img_shape"]),
                          'large_dim': (256, 256, 1),
                          'batch_size': params["batch_size"],
                          'n_channels': 1,
                          'shuffle': True,
                          'fundus_path': "/home/olle/PycharmProjects/thickness_map_prediction/micro_testing/data/fundus_records",
                          'thickness_path': "/home/olle/PycharmProjects/thickness_map_prediction/micro_testing/data/thickness_maps"}

evaluation_params = {'dim': (params["img_shape"][1], params["img_shape"][2]),
                     'batch_size': params["batch_size"],
                     'save_aletoric_record_path': './aleatoric_uncertainty_examples',
                     'save_epistemic_record_path': "./epistemic_uncertainty_examples",
                     "save_aleatoric": True,
                     'save_espistemic': True,
                     'n_channels': 1,
                     'model_path': params["project_dir"] + "fundus_to_thickness_prediction/output_total_variation_filtered_clean_split",
                     'shuffle': True,
                     'test_images': params["data_dir"] + "test_images",
                     'test_labels': params["data_dir"] + "test_labels"}

