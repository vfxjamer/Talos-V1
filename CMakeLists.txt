cmake_minimum_required(VERSION 3.8)
project("Talos" CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA 11.5 needs explicit C++20 flag for nvcc
if(CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++20")
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
