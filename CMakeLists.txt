cmake_minimum_required(VERSION 3.8)
project("Talos" CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# T4 has compute capability 7.5 (Turing) - set explicitly to avoid auto-detection issues
set(CMAKE_CUDA_ARCHITECTURES "75" CACHE STRING "CUDA architectures" FORCE)

# CUDA 11.5's nvcc doesn't support C++20, so use C++17 for CUDA test compiles
# but keep C++20 for host code (needed by RLGymCPP ThreadPool.h concepts)
set(CMAKE_CUDA_STANDARD 17 CACHE STRING "CUDA C++ standard" FORCE)
set(CMAKE_CUDA_STANDARD_REQUIRED ON CACHE BOOL "Require CUDA standard" FORCE)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17" CACHE STRING "CUDA flags" FORCE)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17" CACHE STRING "NVCC flags" FORCE)

# CUDA 13.x removed nvToolsExt (NVTX v1). libtorch's cmake checks for CUDA::nvToolsExt target.
# Create a stub interface target so libtorch's check passes.
if(NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
    if(EXISTS "/usr/local/cuda/include/nvtx3")
        set_property(TARGET CUDA::nvToolsExt PROPERTY INTERFACE_INCLUDE_DIRECTORIES "/usr/local/cuda/include")
    endif()
    if(EXISTS "/usr/local/cuda/lib64/libnvToolsExt.so")
        set_property(TARGET CUDA::nvToolsExt PROPERTY INTERFACE_LINK_LIBRARIES "/usr/local/cuda/lib64/libnvToolsExt.so")
    endif()
    message(STATUS "Created stub CUDA::nvToolsExt target for CUDA 13.x compatibility")
endif()

add_subdirectory(thirdparty/GigaLearnCPP-Leak/GigaLearnCPP)

file(GLOB TALOS_SRC "src/*.cpp" "src/*.h")
list(FILTER TALOS_SRC EXCLUDE REGEX ".*TalosStarter\\.cpp$")

add_executable(Talos ${TALOS_SRC})
if(MSVC)
    target_link_options(Talos PRIVATE /STACK:8388608)
endif()
target_link_libraries(Talos PRIVATE GigaLearnCPP)
target_include_directories(Talos PRIVATE "src")

add_custom_command(TARGET Talos POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_directory
		"$<TARGET_FILE_DIR:GigaLearnCPP>"
		"$<TARGET_FILE_DIR:${PROJECT_NAME}>"
	COMMENT "Copying GigaLearnCPP DLLs to output..."
)

add_custom_command(TARGET Talos POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_directory
		"${CMAKE_SOURCE_DIR}/collision_meshes"
		"$<TARGET_FILE_DIR:${PROJECT_NAME}>/collision_meshes"
	COMMENT "Copying collision meshes to output..."
)

add_custom_command(TARGET Talos POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_directory
		"${CMAKE_SOURCE_DIR}/thirdparty/GigaLearnCPP-Leak/GigaLearnCPP/python_scripts"
		"$<TARGET_FILE_DIR:${PROJECT_NAME}>/python_scripts"
	COMMENT "Copying Python scripts to output..."
)

if(WIN32)
    add_executable(TalosStarter WIN32 src/TalosStarter.cpp)

    add_custom_command(TARGET TalosStarter POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            "$<TARGET_FILE:TalosStarter>"
            "$<TARGET_FILE_DIR:Talos>/"
        COMMENT "Copying TalosStarter to output..."
    )

    add_custom_command(TARGET TalosStarter POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:TalosStarter>"
            "$ENV{APPDATA}/Microsoft/Windows/Start Menu/Programs/Startup/"
        COMMENT "Registering TalosStarter for automatic startup..."
    )
endif()
