# Local libraries
include_directories(${CLM_SOURCE_DIR}/include)
	
include_directories(../../lib/local/CLM/include)
include_directories(../../lib/local/FaceAnalyser/include)
			
add_executable(SimpleCLMImg SimpleCLMImg.cpp)
target_link_libraries(SimpleCLMImg CLM)
target_link_libraries(SimpleCLMImg FaceAnalyser)
target_link_libraries(SimpleCLMImg dlib)

if(WIN32)
	target_link_libraries(SimpleCLMImg ${OpenCVLibraries})
endif(WIN32)
if(UNIX)
    target_link_libraries(SimpleCLMImg ${OpenCV_LIBS} ${Boost_LIBRARIES} ${TBB_LIBRARIES})
endif(UNIX)

install (TARGETS SimpleCLMImg DESTINATION ${CMAKE_BINARY_DIR}/bin)
