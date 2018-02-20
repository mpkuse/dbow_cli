cmake_minimum_required(VERSION 2.8.3)
project(vfc)

if(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
endif()

find_package(OpenCV 3 REQUIRED )
include_directories(
  ${OpenCV_INCLUDE_DIRS}
)


###### DBOW
set( DBOW_DIR "dbow3_src")
include_directories( ${DBOW_DIR}/ )

set( DBOW_HEADER_FILES
        ${DBOW_DIR}/BowVector.h
        ${DBOW_DIR}/Database.h
        ${DBOW_DIR}/DBoW3.h
        ${DBOW_DIR}/DescManip.h
        ${DBOW_DIR}/exports.h
        ${DBOW_DIR}/FeatureVector.h
        ${DBOW_DIR}/QueryResults.h
        ${DBOW_DIR}/quicklz.h
        ${DBOW_DIR}/ScoringObject.h
        ${DBOW_DIR}/timers.h
        ${DBOW_DIR}/Vocabulary.h
)

set( DBOW_SOURCES
        ${DBOW_DIR}/BowVector.cpp
        ${DBOW_DIR}/Database.cpp
        ${DBOW_DIR}/DescManip.cpp
        ${DBOW_DIR}/FeatureVector.cpp
        ${DBOW_DIR}/QueryResults.cpp
        ${DBOW_DIR}/quicklz.c
        ${DBOW_DIR}/ScoringObject.cpp
        ${DBOW_DIR}/Vocabulary.cpp
)

###### END DBOW

add_executable( keypoints keypoints.cpp )
add_executable( dbow3_standalone dbow3_standalone.cpp ${DBOW_SOURCES})

target_link_libraries( keypoints  ${OpenCV_LIBRARIES} )
target_link_libraries( dbow3_standalone  ${OpenCV_LIBRARIES} )
