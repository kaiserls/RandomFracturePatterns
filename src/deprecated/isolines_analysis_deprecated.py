# # Save the isolines as a dict of numpy arrays
#     isolines_path = f"results/isolines/isolines_{data['run']}.npz"
#     isolines_dict = {f"isoline_{i}": isoline for i, isoline in enumerate(isolines)}
#     np.savez(isolines_path, **isolines_dict)
#     analysis_result["isolines"] = isolines_path

#     length = simple_deprecated.crack_length(isolines)
#     analysis_result["length"] = length

#     max_deviation_from_middle = simple_deprecated.max_deviation_from_middle(isolines)
#     analysis_result["max_deviation_from_middle"] = max_deviation_from_middle

#     try:
#         interpolated_isolines = interpolate_isolines(
#             isolines,
#             target_n_points=data["target_n_points"],
#             x_min=data["structured_mesh_min_x"],
#             x_max=data["structured_mesh_max_x"],
#         )
#         interpolated_isolines_path = (
#             f"results/isolines/interpolated_isolines_{data['run']}.npz"
#         )
#         interpolated_isolines_dict = {
#             f"isoline_{i}": isoline for i, isoline in enumerate(interpolated_isolines)
#         }
#         np.savez(interpolated_isolines_path, **interpolated_isolines_dict)
#         analysis_result["interpolated_isolines"] = interpolated_isolines_path

#         width = np.mean(simple_deprecated.crack_width(interpolated_isolines))
#         analysis_result["width"] = width

#         if reference_data is not None:
#             reference_interpolated_isolines_container = np.load(
#                 reference_data["interpolated_isolines"]
#             )
#             reference_interpolated_isolines = [
#                 reference_interpolated_isolines_container[f"isoline_{i}"]
#                 for i in range(len(reference_interpolated_isolines_container.files))
#             ]
#             deviation = np.mean(
#                 simple_deprecated.crack_deviation(
#                     isolines=interpolated_isolines,
#                     reference_isolines=reference_interpolated_isolines,
#                 )
#             )
#             analysis_result["deviation"] = deviation

#     except Exception as e:
#         if verbose:
#             print(
#                 f"Error while measuring run {data['run']} with the following data:\n{data}"
#             )
#             print(f"Expect some analysis results to be None.")
#             # print(f"Error: {e}")
#             # print(''.join(traceback.TracebackException.from_exception(e).format()))
#         else:
#             pass