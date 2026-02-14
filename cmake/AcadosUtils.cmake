# Helper to collect acados generated code
function(collect_acados_generated_code model_prefix output_dir cxx_output_dir)
  # set(output_dir_c ${PROJECT_SOURCE_DIR}/scripts/${model_prefix}/c_generated_code)
  # set(output_dir_cpp ${PROJECT_SOURCE_DIR}/scripts/${model_prefix}/cpp_generated_code)

  if(APPLE)
      set(lib_ext "dylib")
  else()
      set(lib_ext "so")
  endif()


  set(${model_prefix}_SRC_CPP
    ${cxx_output_dir}/casadi_${model_prefix}_internals.cpp
  )

  set(${model_prefix}_HEADERS_CPP
    ${cxx_output_dir}/casadi_${model_prefix}_internals.h
  )

  set(${model_prefix}_HEADERS
    ${output_dir}/acados_solver_${model_prefix}.h
    ${output_dir}/acados_sim_solver_${model_prefix}.h
    ${output_dir}/${model_prefix}_cost/${model_prefix}_cost.h
    ${output_dir}/${model_prefix}_model/${model_prefix}_model.h
  )

  set(${model_prefix}_LIBS
    ${output_dir}/libacados_ocp_solver_${model_prefix}.${lib_ext}
    ${output_dir}/libacados_sim_solver_${model_prefix}.${lib_ext}
  )

  set(${model_prefix}_CASADI_LIBS
    # these casadi libraries are always .so
    ${cxx_output_dir}/libcasadi_${model_prefix}_internals.so
  )
  
  set(${model_prefix}_OUTPUT_FILES 
    ${${model_prefix}_SRC_CPP} 
    # ${${model_prefix}_SRC} 
    ${${model_prefix}_HEADERS}
    ${${model_prefix}_LIBS}
    PARENT_SCOPE
  )

  set(${model_prefix}_SRC "${${model_prefix}_SRC}" PARENT_SCOPE)
  set(${model_prefix}_HEADERS "${${model_prefix}_HEADERS}" PARENT_SCOPE)
  set(${model_prefix}_LIBS "${${model_prefix}_LIBS}" PARENT_SCOPE)
  set(${model_prefix}_CASADI_LIBS "${${model_prefix}_LIBS}" PARENT_SCOPE)

  set(${model_prefix}_SRC_CXX "${${model_prefix}_SRC_CPP}" PARENT_SCOPE)
  set(${model_prefix}_HEADERS_CXX "${${model_prefix}_HEADERS_CPP}" PARENT_SCOPE)

endfunction()


