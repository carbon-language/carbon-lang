# tbi: Configurar ficheros.
configure_file(${llvm_include_path}/llvm/ADT/hash_map.h.in ${llvm_builded_incs_dir}/ADT/hash_map.h)
configure_file(${llvm_include_path}/llvm/ADT/hash_set.h.in ${llvm_builded_incs_dir}/ADT/hash_set.h)
configure_file(${llvm_include_path}/llvm/ADT/iterator.h.in ${llvm_builded_incs_dir}/ADT/iterator.h)
configure_file(${llvm_include_path}/llvm/Support/DataTypes.h.in ${llvm_builded_incs_dir}/Support/DataTypes.h)
configure_file(${llvm_include_path}/llvm/Config/config.h.in ${llvm_builded_incs_dir}/Config/config.h)

file(READ ${llvm_include_path}/../win32/config.h vc_config_text)
file(APPEND ${llvm_builded_incs_dir}/Config/config.h ${vc_config_text})
