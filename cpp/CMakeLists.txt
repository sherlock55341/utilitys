set(TARGET_NAME ${PROJECT_NAME})

set(Boost_USE_STATIC_LIBS ON)
find_package(Boost COMPONENTS system REQUIRED)

file(GLOB CXX_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.c
    )

add_library(${TARGET_NAME}_cxx SHARED ${CXX_SRC_FILES})
target_link_libraries(${TARGET_NAME}_cxx Boost::system)