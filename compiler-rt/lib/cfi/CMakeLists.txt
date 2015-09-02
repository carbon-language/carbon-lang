add_custom_target(cfi)
add_compiler_rt_resource_file(cfi_blacklist cfi_blacklist.txt)
add_dependencies(cfi cfi_blacklist)
add_dependencies(compiler-rt cfi)
