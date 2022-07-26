import sys

sys.dont_write_bytecode = True

params = {}
params["batch_size"] = 1
params["img_shape"] = (256, 256, 1)
params["epochs"] = 40
params["learning_rate"] = 0.001
params["continuing_training"] = False
params["save_path"] = "/home/olle/PycharmProjects/thickness_map_prediction/micro_testing/\
thickness_inference/output_bunet_256_norotate"
params["loss_function"] = "mse_loss"
params["d_rate"] = 0.5
params["project_dir"] = "/home/olle/PycharmProjects/thickness_map_prediction/"
params["data_dir"] = params["project_dir"] + "calculation_thickness_maps/data/thickness_map_data_full_export/"
params["save_predictions_dir"] = "/home/olle/PycharmProjects/thickness_map_prediction/analysis/test_predictions"


#### generator params
gen_params = {'dim': (params["img_shape"][0],params["img_shape"][1]),
          'batch_size': params["batch_size"],
          'n_channels': 1,
          'shuffle': True,
          'fundus_path': params["data_dir"] + "fundus",
          'thickness_path': params["data_dir"] + "thickness_maps"}

multi_input_gen_params = {'small_dim': (params["img_shape"]),
                          'large_dim': (256,256,1),
          'batch_size': params["batch_size"],
          'n_channels': 1,
          'shuffle': True,
          'fundus_path':"/home/olle/PycharmProjects/thickness_map_prediction/micro_testing/data/fundus_records",
          'thickness_path':  "/home/olle/PycharmProjects/thickness_map_prediction/micro_testing/data/thickness_maps"}

evaluation_params = {'dim': (params["img_shape"][1],params["img_shape"][2]),
          'batch_size': params["batch_size"],
          'save_aletoric_record_path':'./aleatoric_uncertainty_examples',
          'save_epistemic_record_path': "./epistemic_uncertainty_examples",
          "save_aleatoric": True,
          'save_espistemic': True,
          'n_channels': 1,
          'model_path':params["project_dir"] +"fundus_to_thickness_prediction/output_total_variation_filtered_clean_split",
          'shuffle': True,
          'test_images':  params["data_dir"] + "test_images",
          'test_labels':  params["data_dir"] + "test_labels"}


#'fundus_path':"/home/icb/olle.holmberg/projects/data/thickness_map_data/fundus",
#'thickness_path':"/home/icb/olle.holmberg/projects/data/thickness_map_data/thickness_maps"
