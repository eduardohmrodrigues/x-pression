///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2014, University of Southern California and University of Cambridge,
// all rights reserved.
//
// THIS SOFTWARE IS PROVIDED “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES,
// INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
// THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY. OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Notwithstanding the license granted herein, Licensee acknowledges that certain components
// of the Software may be covered by so-called “open source” software licenses (“Open Source
// Components”), which means any software licenses approved as open source licenses by the
// Open Source Initiative or any substantially similar licenses, including without limitation any
// license that, as a condition of distribution of the software licensed under such license,
// requires that the distributor make the software available in source code format. Licensor shall
// provide a list of Open Source Components for a particular version of the Software upon
// Licensee’s request. Licensee will comply with the applicable terms of such licenses and to
// the extent required by the licenses covering Open Source Components, the terms of such
// licenses will apply in lieu of the terms of this Agreement. To the extent the terms of the
// licenses applicable to Open Source Components prohibit any of the restrictions in this
// License Agreement with respect to such Open Source Component, such restrictions will not
// apply to such Open Source Component. To the extent the terms of the licenses applicable to
// Open Source Components require Licensor to make an offer to provide source code or
// related information in connection with the Software, such offer is hereby made. Any request
// for source code or related information should be directed to cl-face-tracker-distribution@lists.cam.ac.uk
// Licensee acknowledges receipt of notices for the Open Source Components for the initial
// delivery of the Software.

//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite one of the following works:
//
//       Tadas Baltrusaitis, Peter Robinson, and Louis-Philippe Morency. 3D
//       Constrained Local Model for Rigid and Non-Rigid Facial Tracking.
//       IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2012.    
//
//       Tadas Baltrusaitis, Peter Robinson, and Louis-Philippe Morency. 
//       Constrained Local Neural Fields for robust facial landmark detection in the wild.
//       in IEEE Int. Conference on Computer Vision Workshops, 300 Faces in-the-Wild Challenge, 2013.    
//
///////////////////////////////////////////////////////////////////////////////

// SimpleCLM.cpp : Defines the entry point for the console application.
#include "CLM_core.h"

#include <fstream>
#include <sstream>
#include <ctime> // finding time
#include <chrono> // time
#include <algorithm> //min

#include <opencv2/videoio/videoio.hpp>  // Video write
#include <opencv2/videoio/videoio_c.h>  // Video write

enum EEmotion{
	NEUTRAL,
	HAPPY,
	SAD,
	ANGRY,
	SCARED,
	SURPRISED,
	DOUBT,
	UNKNOWN
};

struct eye_struct{
	double eye_height;
};

struct eyebrow_struct{
	double eyebrow_height;
	double eyebrow2eye_height;
};

struct mouth_struct{
	double mouth_width;
	double mouth_height;
};

struct face_struct{
	eye_struct leftEye, rightEye;
	eyebrow_struct leftEyebrow, rightEyebrow;
	mouth_struct mouth;
	EEmotion emotion;
};

#define PI 3.1415926535897
#define CALCULATE_POSE false
#define KNN 0 //1 - Usando KNN, 0 - Usando matriz dissimilaridade
int neighbours = 10;

vector<Point2i> actualLandmarks;

#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl

void init();

void showImageWithLandMarks(Mat capturedImage, vector<Point2i> landMarks);
vector<Point2i> getNormalizedLandMarks(Mat capturedImage, vector<Point2i> landMarks);
Point2i rotate_point(Point2i center, Point2i p, float angle);
Point2i scale_point(Point2i center, Point2i p, float scale);

face_struct		getFeatureFace(vector<Point2i> landMarks);

vector<Point2i> extractLandMarks(vector<Point2i> landMarks, int index_start, int index_end);

vector<Point2i> getLeftEye(vector<Point2i> landMarks);
vector<Point2i> getRightEye(vector<Point2i> landMarks);
vector<Point2i> getLeftEyebrow(vector<Point2i> landMarks);
vector<Point2i> getRightEyebrow(vector<Point2i> landMarks);
vector<Point2i> getMouth(vector<Point2i> landMarks);

eye_struct		getFeatureEye(vector<Point2i> eyeLandmarks);
eyebrow_struct	getFeatureEyebrow(vector<Point2i> eyebrowLandmarks, vector<Point2i> eyeLandmarks);
mouth_struct	getFeatureMouth(vector<Point2i> mouthLandmarks);

double distanceLineToPoint(vector<Point2i> line, Point2i point);

double getHeightEye(vector<Point2i> eyeLandmarks);
double getHeightEyebrow(vector<Point2i> eyebrowLandmarks);
double getHeightMouth(vector<Point2i> mouthLandmarks);
double getWidthMouth(vector<Point2i> mouthLandmarks);
double getHeightEyebrow2Eye(vector<Point2i> eyeLandmarks, vector<Point2i> eyebrowLandmarks);

EEmotion getEmotion(face_struct actualFace);
double getDistance(face_struct actualFace, face_struct baseFace);
map<string, double> distanceGambiarra;

double getEyeDistance(eye_struct actualEye, eye_struct baseEye);
double getEyebrowDistance(eyebrow_struct actualEyebrow, eyebrow_struct baseEyebrow);
double getMouthDistance(mouth_struct actualMouth, mouth_struct baseMouth);

vector<double> featureSnapshot(face_struct currentFace);

void populateFaces();
void exportFaces();

void inputFace(face_struct currentFace, EEmotion emotion);

//FOR INPUTTING FACES ONLY
// i -> Activate input mode
// a -> Add said face with the emotion of choice
// s -> Next emotion
bool inputMode = false;
face_struct newFace = face_struct{};
vector<EEmotion> emotionVec;
int choiceEmotion = 0;


//CAPTURE MODE
bool captureMode = false;
time_t sessionTime;
ofstream outputCSV;
string getSessionName();
string getSessionTime();
clock_t captureStart;
int captureRate = 1;
int tickCount = 0;


void pushFaceToFile(face_struct currentFace);

vector<face_struct> facePopulation;

map<EEmotion, string> emotionMap;

void init(){
	emotionMap[EEmotion::UNKNOWN] = "Unknown";
	emotionMap[EEmotion::NEUTRAL] = "Neutral";
	emotionMap[EEmotion::HAPPY] = "Happy";
	emotionMap[EEmotion::SAD] = "Sad";
	emotionMap[EEmotion::ANGRY] = "Angry";
	emotionMap[EEmotion::SCARED] = "Scared";
	emotionMap[EEmotion::SURPRISED] = "Surprised";
	emotionMap[EEmotion::DOUBT] = "Doubt";

	emotionVec.clear();

	emotionVec.push_back(EEmotion::NEUTRAL);
	emotionVec.push_back(EEmotion::HAPPY);
	emotionVec.push_back(EEmotion::SAD);
	emotionVec.push_back(EEmotion::ANGRY);
	emotionVec.push_back(EEmotion::SCARED);
	emotionVec.push_back(EEmotion::SURPRISED);
	emotionVec.push_back(EEmotion::DOUBT);
	
	//Populate Faces
	populateFaces();
}

static void printErrorAndAbort(const std::string & error)
{
	std::cout << error << std::endl;
	abort();
}

#define FATAL_STREAM( stream ) \
printErrorAndAbort( std::string( "Fatal error: " ) + stream )

using namespace std;
using namespace cv;

vector<string> get_arguments(int argc, char **argv)
{

	vector<string> arguments;

	for (int i = 0; i < argc; ++i)
	{
		arguments.push_back(string(argv[i]));
	}
	return arguments;
}

// Some globals for tracking timing information for visualisation
double fps_tracker = -1.0;
int64 t0 = 0;

// Visualising the results
void visualise_tracking(Mat& captured_image, Mat_<float>& depth_image, const CLMTracker::CLM& clm_model, const CLMTracker::CLMParameters& clm_parameters, int frame_count, double fx, double fy, double cx, double cy, vector<Point2i>* landMarks)
{

	// Drawing the facial landmarks on the face and the bounding box around it if tracking is successful and initialised
	double detection_certainty = clm_model.detection_certainty;
	bool detection_success = clm_model.detection_success;

	double visualisation_boundary = 0.2;

	// Only draw if the reliability is reasonable, the value is slightly ad-hoc
	if (detection_certainty < visualisation_boundary)
	{
		CLMTracker::Draw(captured_image, clm_model, landMarks);

		double vis_certainty = detection_certainty;
		if (vis_certainty > 1)
			vis_certainty = 1;
		if (vis_certainty < -1)
			vis_certainty = -1;

		vis_certainty = (vis_certainty + 1) / (visualisation_boundary + 1);

		if (CALCULATE_POSE){
			Vec6d pose_estimate_to_draw;
			// A rough heuristic for box around the face width
			int thickness = (int)std::ceil(2.0* ((double)captured_image.cols) / 640.0);

			pose_estimate_to_draw = CLMTracker::GetCorrectedPoseWorld(clm_model, fx, fy, cx, cy);

			// Draw it in reddish if uncertain, blueish if certain
			CLMTracker::DrawBox(captured_image, pose_estimate_to_draw, Scalar((1 - vis_certainty)*255.0, 0, vis_certainty * 255), thickness, fx, fy, cx, cy);
		}

	}

	// Work out the framerate
	if (frame_count % 10 == 0)
	{
		double t1 = cv::getTickCount();
		fps_tracker = 10.0 / (double(t1 - t0) / cv::getTickFrequency());
		t0 = t1;
	}

	// Write out the framerate on the image before displaying it
	char fpsC[255];
	std::sprintf(fpsC, "%d", (int)fps_tracker);
	string fpsSt("FPS:");
	fpsSt += fpsC;
	cv::putText(captured_image, fpsSt, cv::Point(10, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 0, 0));

	//if (!clm_parameters.quiet_mode)
	//{
	//	namedWindow("tracking_result", 1);
	//	imshow("tracking_result", captured_image);
	//
	//	if (!depth_image.empty())
	//	{
	//		// Division needed for visualisation purposes
	//		imshow("depth", depth_image / 2000.0);
	//	}
	//
	//}
}

int main(int argc, char **argv)
{
	init();



	vector<string> arguments = get_arguments(argc, argv);

	// Some initial parameters that can be overriden from command line	
	vector<string> files, depth_directories, pose_output_files, tracked_videos_output, landmark_output_files, landmark_3D_output_files;

	// By default try webcam 0
	int device = 0;



	CLMTracker::CLMParameters clm_parameters(arguments);

	// Get the input output file parameters

	// Indicates that rotation should be with respect to world or camera coordinates
	bool use_world_coordinates;
	CLMTracker::get_video_input_output_params(files, depth_directories, pose_output_files, tracked_videos_output, landmark_output_files, landmark_3D_output_files, use_world_coordinates, arguments);

	// The modules that are being used for tracking
	CLMTracker::CLM clm_model(clm_parameters.model_location);

	// Grab camera parameters, if they are not defined (approximate values will be used)
	float fx = 0, fy = 0, cx = 0, cy = 0;
	// Get camera parameters
	CLMTracker::get_camera_params(device, fx, fy, cx, cy, arguments);

	// If cx (optical axis centre) is undefined will use the image size/2 as an estimate
	bool cx_undefined = false;
	bool fx_undefined = false;
	if (cx == 0 || cy == 0)
	{
		cx_undefined = true;
	}
	if (fx == 0 || fy == 0)
	{
		fx_undefined = true;
	}

	// If multiple video files are tracked, use this to indicate if we are done
	bool done = false;
	int f_n = -1;

	while (!done) // this is not a for loop as we might also be reading from a webcam
	{

		string current_file;

		// We might specify multiple video files as arguments
		if (files.size() > 0)
		{
			f_n++;
			current_file = files[f_n];
		}
		else
		{
			// If we want to write out from webcam
			f_n = 0;
		}

		double fps_vid_in = -1.0;

		bool use_depth = !depth_directories.empty();

		// Do some grabbing
		VideoCapture video_capture;

		INFO_STREAM("Attempting to capture from device: " << device);
		video_capture = VideoCapture(device);

		//if opened the camera
		if (video_capture.isOpened()){
			// Read a first frame often empty in camera
			Mat captured_image;
			video_capture >> captured_image;
		}
		else{
			INFO_STREAM("Attempting to capture from device: " << device+1);
			video_capture.open(device+1);
		}
		
		if (!video_capture.isOpened() && current_file.size() > 0)
		{
			INFO_STREAM("Attempting to read from file: " << current_file);
			video_capture = VideoCapture(current_file);
			fps_vid_in = video_capture.get(CV_CAP_PROP_FPS);

			// Check if fps is nan or less than 0
			if (fps_vid_in != fps_vid_in || fps_vid_in <= 0)
			{
				INFO_STREAM("FPS of the video file cannot be determined, assuming 30");
				fps_vid_in = 30;
			}
		}

		if (!video_capture.isOpened()) FATAL_STREAM("Failed to open video source");
		else INFO_STREAM("Device or file opened");

		Mat captured_image;
		video_capture >> captured_image;

		// If optical centers are not defined just use center of image
		if (cx_undefined)
		{
			cx = captured_image.cols / 2.0f;
			cy = captured_image.rows / 2.0f;
		}
		// Use a rough guess-timate of focal length
		if (fx_undefined)
		{
			fx = 500 * (captured_image.cols / 640.0);
			fy = 500 * (captured_image.rows / 480.0);

			fx = (fx + fy) / 2.0;
			fy = fx;
		}

		// Creating output files
		std::ofstream pose_output_file;
		if (!pose_output_files.empty())
		{
			pose_output_file.open(pose_output_files[f_n], ios_base::out);
			pose_output_file << "frame, timestamp, confidence, success, Tx, Ty, Tz, Rx, Ry, Rz";
			pose_output_file << endl;
		}

		std::ofstream landmarks_output_file;
		if (!landmark_output_files.empty())
		{
			landmarks_output_file.open(landmark_output_files[f_n], ios_base::out);
			landmarks_output_file << "frame, timestamp, confidence, success";
			for (int i = 0; i < clm_model.pdm.NumberOfPoints(); ++i)
				landmarks_output_file << ", x" << i;

			for (int i = 0; i < clm_model.pdm.NumberOfPoints(); ++i)
				landmarks_output_file << ", y" << i;

			landmarks_output_file << endl;
		}

		std::ofstream landmarks_3D_output_file;
		if (!landmark_3D_output_files.empty())
		{
			landmarks_3D_output_file.open(landmark_3D_output_files[f_n], ios_base::out);

			landmarks_3D_output_file << "frame, timestamp, confidence, success";
			for (int i = 0; i < clm_model.pdm.NumberOfPoints(); ++i)
				landmarks_3D_output_file << ", X" << i;

			for (int i = 0; i < clm_model.pdm.NumberOfPoints(); ++i)
				landmarks_3D_output_file << ", Y" << i;

			for (int i = 0; i < clm_model.pdm.NumberOfPoints(); ++i)
				landmarks_3D_output_file << ", Z" << i;

			landmarks_3D_output_file << endl;
		}

		int frame_count = 0;

		// saving the videos
		VideoWriter writerFace;
		if (!tracked_videos_output.empty())
		{
			double fps = fps_vid_in == -1 ? 30 : fps_vid_in;
			writerFace = VideoWriter(tracked_videos_output[f_n], CV_FOURCC('D', 'I', 'V', 'X'), fps, captured_image.size(), true);
		}

		// Use for timestamping if using a webcam
		int64 t_initial = cv::getTickCount();

		// Timestamp in seconds of current processing
		double time_stamp = 0;

		INFO_STREAM("Starting tracking");
		while (!captured_image.empty())
		{

			// Grab the timestamp first
			if (fps_vid_in == -1)
			{
				int64 curr_time = cv::getTickCount();
				time_stamp = (double(curr_time - t_initial) / cv::getTickFrequency());
			}
			else
			{
				time_stamp = (double)frame_count * (1.0 / fps_vid_in);
			}

			// Reading the images
			Mat_<float> depth_image;
			Mat_<uchar> grayscale_image;

			if (captured_image.channels() == 3)
			{
				cvtColor(captured_image, grayscale_image, CV_BGR2GRAY);
			}
			else
			{
				grayscale_image = captured_image.clone();
			}

			// Get depth image
			if (use_depth)
			{
				char* dst = new char[100];
				std::stringstream sstream;

				sstream << depth_directories[f_n] << "\\depth%05d.png";
				sprintf(dst, sstream.str().c_str(), frame_count + 1);
				// Reading in 16-bit png image representing depth
				Mat_<short> depth_image_16_bit = imread(string(dst), -1);

				// Convert to a floating point depth image
				if (!depth_image_16_bit.empty())
				{
					depth_image_16_bit.convertTo(depth_image, CV_32F);
				}
				else
				{
					WARN_STREAM("Can't find depth image");
				}
			}

			// The actual facial landmark detection / tracking
			bool detection_success = CLMTracker::DetectLandmarksInVideo(grayscale_image, depth_image, clm_model, clm_parameters);

			// Work out the pose of the head from the tracked model
			Vec6d pose_estimate_CLM;
			if (CALCULATE_POSE){
				if (use_world_coordinates)
				{
					pose_estimate_CLM = CLMTracker::GetCorrectedPoseWorld(clm_model, fx, fy, cx, cy);
				}
				else
				{
					pose_estimate_CLM = CLMTracker::GetCorrectedPoseCamera(clm_model, fx, fy, cx, cy);
				}
			}

			// Visualising the results
			// Drawing the facial landmarks on the face and the bounding box around it if tracking is successful and initialised
			double detection_certainty = clm_model.detection_certainty;

			visualise_tracking(captured_image, depth_image, clm_model, clm_parameters, frame_count, fx, fy, cx, cy, &actualLandmarks);
			showImageWithLandMarks(captured_image, actualLandmarks);
			actualLandmarks.clear();

			// Output the detected facial landmarks
			if (!landmark_output_files.empty())
			{
				double confidence = 0.5 * (1 - clm_model.detection_certainty);
				landmarks_output_file << frame_count + 1 << ", " << time_stamp << ", " << confidence << ", " << detection_success;
				for (int i = 0; i < clm_model.pdm.NumberOfPoints() * 2; ++i)
				{
					landmarks_output_file << ", " << clm_model.detected_landmarks.at<double>(i);
				}
				landmarks_output_file << endl;
			}

			// Output the detected facial landmarks
			if (!landmark_3D_output_files.empty())
			{
				double confidence = 0.5 * (1 - clm_model.detection_certainty);
				landmarks_3D_output_file << frame_count + 1 << ", " << time_stamp << ", " << confidence << ", " << detection_success;
				Mat_<double> shape_3D = clm_model.GetShape(fx, fy, cx, cy);
				for (int i = 0; i < clm_model.pdm.NumberOfPoints() * 3; ++i)
				{
					landmarks_3D_output_file << ", " << shape_3D.at<double>(i);
				}
				landmarks_3D_output_file << endl;
			}

			// Output the estimated head pose
			if (!pose_output_files.empty() && CALCULATE_POSE)
			{
				double confidence = 0.5 * (1 - clm_model.detection_certainty);
				pose_output_file << frame_count + 1 << ", " << time_stamp << ", " << confidence << ", " << detection_success
					<< ", " << pose_estimate_CLM[0] << ", " << pose_estimate_CLM[1] << ", " << pose_estimate_CLM[2]
					<< ", " << pose_estimate_CLM[3] << ", " << pose_estimate_CLM[4] << ", " << pose_estimate_CLM[5] << endl;
			}

			// output the tracked video
			if (!tracked_videos_output.empty())
			{
				writerFace << captured_image;
			}

			video_capture >> captured_image;

			cv::flip(captured_image, captured_image, 1);

			// detect key presses
			char character_press = cv::waitKey(1);

			// restart the tracker
			if (character_press == 'r')
			{
				clm_model.Reset();
			}
			// quit the application
			else if (character_press == 'q')
			{
				return(0);
			}
			else if (character_press == 'i')
			{
				inputMode = !inputMode;
			}
			else if (character_press == 'a' && inputMode)
			{
				inputFace(newFace, emotionVec.at(choiceEmotion));
			}
			else if (character_press == 's' && inputMode)
			{
				if (choiceEmotion != emotionVec.size()-1)
				{
					choiceEmotion++;
				}
				else{
					choiceEmotion = 0;
				}
			}
			else if (character_press == 'd' && inputMode)
			{
				exportFaces();
			}
			else if (character_press == 'c')
			{
				if (captureMode == true)
				{
					captureMode = false;
					outputCSV.close();
				}
				else if (captureMode == false){
					captureMode = true;
					string sessionName = getSessionName();
					captureStart = clock();
					outputCSV.open(sessionName);
					outputCSV << "Expression,Left Eye Height,Right Eye Height,Left Eyebrow Height,Right Eyebrow Height,LeftEyebrow to Eye Distance,RightEyebrow to Eye Distance,Mouth Width,Mouth Height,Timestamp\n";
				}
			}


			// Update the frame count
			frame_count++;

		}

		frame_count = 0;

		// Reset the model, for the next video
		clm_model.Reset();

		pose_output_file.close();
		landmarks_output_file.close();

		// break out of the loop if done with all the files (or using a webcam)
		if (f_n == files.size() - 1 || files.empty())
		{
			done = true;
		}
	}

	return 0;
}

void showImageWithLandMarks(Mat capturedImage, vector<Point2i> landMarks){
	int thickness = (int)std::ceil(5.0* ((double)capturedImage.cols) / 640.0);
	int thickness_2 = (int)std::ceil(1.5* ((double)capturedImage.cols) / 640.0);

	Mat img = capturedImage.clone();
	 
	Point2i featurePoint;
	//for (int i = 0; i < landMarks.size(); i++){
	//	featurePoint.x = landMarks.at(i).x;
	//	featurePoint.y = landMarks.at(i).y;
	//
	//	cv::circle(img, featurePoint, 1, Scalar(0, 0, 255), thickness);
	//	cv::circle(img, featurePoint, 1, Scalar(255, 0, 0), thickness_2);
	//
	//	cv::putText(img, to_string(i), featurePoint, CV_FONT_NORMAL, 0.3, Scalar(0, 0, 255));
	//}

	vector<Point2i> normalizedLandMarks = getNormalizedLandMarks(capturedImage, landMarks);

	if (!landMarks.empty()){
		face_struct actualFace = getFeatureFace(normalizedLandMarks);


		//DEBUG PURPOSES
		int initY = 100;
		int dist = 20;

		//vector<double> distances = featureSnapshot(actualFace);

		//cv::putText(img, "Left Eye H.: " + to_string(distances[0])		, Point2i(20, initY + 1 * dist), CV_FONT_NORMAL, 0.3, Scalar(0, 0, 255));
		//cv::putText(img, "Right Eye H.: " + to_string(distances[1])		, Point2i(20, initY + 2 * dist), CV_FONT_NORMAL, 0.3, Scalar(0, 0, 255));
		//cv::putText(img, "Left Eyebrow H.: " + to_string(distances[2])	, Point2i(20, initY + 3 * dist), CV_FONT_NORMAL, 0.3, Scalar(0, 0, 255));
		//cv::putText(img, "Right Eyebrow H.: " + to_string(distances[3])	, Point2i(20, initY + 4 * dist), CV_FONT_NORMAL, 0.3, Scalar(0, 0, 255));
		//cv::putText(img, "Mouth H.: " + to_string(distances[4])			, Point2i(20, initY + 5 * dist), CV_FONT_NORMAL, 0.3, Scalar(0, 0, 255));

		actualFace.emotion = getEmotion(actualFace);

		double totalDistance = 0;

		for (int i = 0; i < emotionVec.size(); i++)
			totalDistance += distanceGambiarra[emotionMap[emotionVec[i]]];

			#if KNN
				cv::putText(img, "Happy:     " + to_string(100 - (distanceGambiarra[emotionMap[EEmotion::HAPPY]] * 100 / totalDistance)) + "%", Point2i(20, initY + 6 * dist), CV_FONT_NORMAL, 0.5, Scalar(0, 0, 255));			 
				cv::putText(img, "Sad:       " + to_string(100 - (distanceGambiarra[emotionMap[EEmotion::SAD]] * 100 / totalDistance)) + "%", Point2i(20, initY + 7 * dist), CV_FONT_NORMAL, 0.5, Scalar(0, 0, 255));			 
				cv::putText(img, "Surprised: " + to_string(100 - (distanceGambiarra[emotionMap[EEmotion::SURPRISED]] * 100 / totalDistance)) + "%", Point2i(20, initY + 8 * dist), CV_FONT_NORMAL, 0.5, Scalar(0, 0, 255));		 
				cv::putText(img, "Angry:     " + to_string(100 - (distanceGambiarra[emotionMap[EEmotion::ANGRY]] * 100 / totalDistance)) + "%", Point2i(20, initY + 9 * dist), CV_FONT_NORMAL, 0.5, Scalar(0, 0, 255));			 
				cv::putText(img, "Scared:    " + to_string(100 - (distanceGambiarra[emotionMap[EEmotion::SCARED]] * 100 / totalDistance)) + "%", Point2i(20, initY + 10 * dist), CV_FONT_NORMAL, 0.5, Scalar(0, 0, 255));		 
				cv::putText(img, "Neutral:   " + to_string(100 - (distanceGambiarra[emotionMap[EEmotion::NEUTRAL]] * 100 / totalDistance)) + "%", Point2i(20, initY + 11 * dist), CV_FONT_NORMAL, 0.5, Scalar(0, 0, 255));		 

				cv::putText(img, "Happy:     " + to_string(roundf((double)distanceGambiarra[emotionMap[EEmotion::HAPPY]])) + "%", Point2i(20, initY + 6 * dist), CV_FONT_NORMAL, 0.5, Scalar(0, 0, 255));
				cv::putText(img, "Sad:       " + to_string(roundf((double)distanceGambiarra[emotionMap[EEmotion::SAD]])) + "%", Point2i(20, initY + 7 * dist), CV_FONT_NORMAL, 0.5, Scalar(0, 0, 255));
				cv::putText(img, "Surprised: " + to_string(roundf(distanceGambiarra[emotionMap[EEmotion::SURPRISED]])) + "%", Point2i(20, initY + 8 * dist), CV_FONT_NORMAL, 0.5, Scalar(0, 0, 255));
				cv::putText(img, "Angry:     " + to_string(roundf(distanceGambiarra[emotionMap[EEmotion::ANGRY]])) + "%", Point2i(20, initY + 9 * dist), CV_FONT_NORMAL, 0.5, Scalar(0, 0, 255));
				cv::putText(img, "Scared:    " + to_string(roundf(distanceGambiarra[emotionMap[EEmotion::SCARED]])) + "%", Point2i(20, initY + 10 * dist), CV_FONT_NORMAL, 0.5, Scalar(0, 0, 255));
				cv::putText(img, "Neutral:   " + to_string(roundf(distanceGambiarra[emotionMap[EEmotion::NEUTRAL]])) + "%", Point2i(20, initY + 11 * dist), CV_FONT_NORMAL, 0.5, Scalar(0, 0, 255));
			//#else
			//string aux;
			//int nowSize;
			//for (int i = 0; i < emotionVec.size(); i++){
			//	aux = emotionMap[emotionVec[i]];
			//	nowSize = aux.length();
			//	for (int i = 0; i < 15 - nowSize; i++)
			//		aux.append(" ");
			//	aux.append(":");
			//
			//	cv::putText(img, aux + to_string(100 - (distanceGambiarra[emotionMap[emotionVec[i]]] * 100 / totalDistance)) + "%", Point2i(20, initY + ((i+6) * dist)), CV_FONT_NORMAL, 0.5, Scalar(0, 0, 255));
			//}
			#endif


		cv::putText(img, "EMOTION: " + emotionMap[actualFace.emotion], Point2i(10, initY), CV_FONT_NORMAL, 0.8, Scalar(0, 0, 255), 1.5f);
		cv::putText(img, "Press I to calibrate ",                 Point2i(10, initY + 40), CV_FONT_NORMAL, 0.8, Scalar(0, 0, 255), 1.5f);

		if (inputMode)
		{
			string emotionSelectionList = "";

			for (int i = 0; i < emotionVec.size(); i++){
				emotionSelectionList.append("|");
				emotionSelectionList.append(to_string(i));
				emotionSelectionList.append("-");
				emotionSelectionList.append(emotionMap[emotionVec.at(i)]);
			}

			newFace = actualFace;

			int lineBeginY = 340;
			int breaklineSpace = 20;
			cv::putText(img, "INPUT MODE", Point2i(20, (lineBeginY + 0*breaklineSpace)), CV_FONT_NORMAL, 0.6, Scalar(255, 0, 0));
			cv::putText(img, "Inputing Face as emotion: " + to_string(choiceEmotion), Point2i(20, (lineBeginY + 1*breaklineSpace)), CV_FONT_NORMAL, 0.5, Scalar(255, 0, 0));
			cv::putText(img, emotionSelectionList, Point2i(20, (lineBeginY + 2*breaklineSpace)), CV_FONT_NORMAL, 0.5, Scalar(255, 0, 0));
			cv::putText(img, "a - Adds current face. s - Cycle through emotions d - Export faces", Point2i(20, (lineBeginY + 3*breaklineSpace)), CV_FONT_NORMAL, 0.5, Scalar(255, 0, 0));
		}

		if (captureMode)
		{
			tickCount += 1;
			cv::putText(img, "CAPTURE MODE ON", Point2i(10, 40), CV_FONT_NORMAL, 1.0, Scalar(125, 255, 125));
			if (tickCount >= captureRate){
				pushFaceToFile(actualFace);
				tickCount = 0;
			}
		}
		else{
			cv::putText(img, "CAPTURE MODE OFF", Point2i(10, 40), CV_FONT_NORMAL, 1.0, Scalar(125, 125, 255));
		}
	}

	imshow("LandMarks Vector", img);
}

vector<Point2i> getNormalizedLandMarks(Mat capturedImage, vector<Point2i> landMarks){
	vector<Point2i> normalizedLandmarks;
	normalizedLandmarks.clear();
	if (landMarks.size() > 0){
		Mat img2 = Mat::ones(capturedImage.size(), capturedImage.type());
		int thickness = (int)std::ceil(5.0* ((double)capturedImage.cols) / 640.0);
		int thickness_2 = (int)std::ceil(1.5* ((double)capturedImage.cols) / 640.0);

		//face começa em 0 e termina em 16
		int catetoOposto, catetoAdjacente, hipotenusa, eyeDistance;
		double sen, arcosenoRad, arcosenoDegree;
		catetoOposto = abs(landMarks[0].y - landMarks[16].y);
		catetoAdjacente = abs(landMarks[16].x - landMarks[0].x);
		hipotenusa = sqrt(pow(catetoOposto, 2) + pow(catetoAdjacente, 2));
		//if do not cast at least one of the arguments to double the division always returns 0
		sen = ((double)catetoOposto / (double)hipotenusa);
		arcosenoRad = asin(sen);
		arcosenoDegree = arcosenoRad * 180 / PI;

		eyeDistance = landMarks[45].x - landMarks[36].x;

		int firstLineY = 30;
		int linePxSize = 20;

		string line0 = "Cateto Oposto: ";
		line0.append(to_string(catetoOposto));
		string line1 = "Cateto Adjacente: ";
		line1.append(to_string(catetoAdjacente));
		string line2 = "Hipotenusa: ";
		line2.append(to_string(hipotenusa));
		string line3 = "Seno: ";
		line3.append(to_string(sen));
		string line4 = "Arcoseno: ";
		line4.append(to_string(arcosenoDegree));
		string line5 = "36 to 45: ";
		line5.append(to_string(eyeDistance));

		cv::putText(img2, line0, Point(0, firstLineY + 0 * linePxSize), CV_FONT_NORMAL, 0.5, Scalar(0, 0, 255));
		cv::putText(img2, line1, Point(0, firstLineY + 1 * linePxSize), CV_FONT_NORMAL, 0.5, Scalar(0, 0, 255));
		cv::putText(img2, line2, Point(0, firstLineY + 2 * linePxSize), CV_FONT_NORMAL, 0.5, Scalar(0, 0, 255));
		cv::putText(img2, line3, Point(0, firstLineY + 3 * linePxSize), CV_FONT_NORMAL, 0.5, Scalar(0, 0, 255));
		cv::putText(img2, line4, Point(0, firstLineY + 4 * linePxSize), CV_FONT_NORMAL, 0.5, Scalar(0, 0, 255));
		cv::putText(img2, line5, Point(0, firstLineY + 5 * linePxSize), CV_FONT_NORMAL, 0.5, Scalar(0, 0, 255));

		//nariz = 33
		Point2i featurePoint, normalizedFeaturePoint;
		int deslocationX = (capturedImage.cols / 2) - landMarks[33].x;
		int deslocationY = (capturedImage.rows / 2) - landMarks[33].y;

		//160 boa distancia entre 36 e 45
		double defaultEyeDistance = 160;
		double scaleValue = defaultEyeDistance / eyeDistance;

		for (int i = 0; i < landMarks.size(); i++){
			featurePoint.x = landMarks.at(i).x;
			featurePoint.y = landMarks.at(i).y;

			//first we eliminate the rotation of the face
			if (landMarks[16].y < landMarks[0].y)
				normalizedFeaturePoint = rotate_point(landMarks[33], featurePoint, arcosenoRad);
			else
				normalizedFeaturePoint = rotate_point(landMarks[33], featurePoint, -arcosenoRad);

			//the we scale the face to our stable face size
			normalizedFeaturePoint = scale_point(landMarks[33], normalizedFeaturePoint, scaleValue);

			//then we centralize the face on the image
			normalizedFeaturePoint.x += deslocationX;
			normalizedFeaturePoint.y += deslocationY;

			normalizedLandmarks.push_back(normalizedFeaturePoint);

			cv::circle(img2, normalizedFeaturePoint, 1, Scalar(0, 0, 255), thickness);
			cv::circle(img2, normalizedFeaturePoint, 1, Scalar(255, 0, 0), thickness_2);

			cv::putText(img2, to_string(i), normalizedFeaturePoint, CV_FONT_NORMAL, 0.3, Scalar(0, 0, 255));

			imshow("Normalized Face", img2);


		}
	}

	return normalizedLandmarks;
}

Point2i rotate_point(Point2i center, Point2i p, float angle)
{
	float s = sin(angle);
	float c = cos(angle);

	// translate point back to origin:
	p.x -= center.x;
	p.y -= center.y;

	// rotate point
	float xnew = p.x * c - p.y * s;
	float ynew = p.x * s + p.y * c;

	// translate point back:
	p.x = xnew + center.x;
	p.y = ynew + center.y;
	return p;
}

Point2i scale_point(Point2i center, Point2i p, float scale){
	// translate point back to origin:
	p.x -= center.x;
	p.y -= center.y;

	// translate point
	float xnew = p.x * scale;
	float ynew = p.y * scale;

	// translate point back:
	p.x = xnew + center.x;
	p.y = ynew + center.y;
	return p;
}

face_struct	getFeatureFace(vector<Point2i> landMarks){


	vector<Point2i> leftEyeLandmarks = getLeftEye(landMarks);
	vector<Point2i> rightEyeLandmarks = getRightEye(landMarks);
	vector<Point2i> leftEyebrowLandmarks = getLeftEyebrow(landMarks);
	vector<Point2i> rightEyebrowLandmarks = getRightEyebrow(landMarks);
	vector<Point2i> mouthLandmarks = getMouth(landMarks);

	eye_struct		leftEyeStruct = getFeatureEye(leftEyeLandmarks);
	eye_struct		rightEyeStruct = getFeatureEye(rightEyeLandmarks);
	eyebrow_struct	leftEyebrowStruct = getFeatureEyebrow(leftEyebrowLandmarks, leftEyeLandmarks);
	eyebrow_struct	rightEyebrowStruct = getFeatureEyebrow(rightEyebrowLandmarks, rightEyeLandmarks);
	mouth_struct	mouthStruct = getFeatureMouth(mouthLandmarks);

	face_struct currentFace =
	{
		leftEyeStruct,
		rightEyeStruct,
		leftEyebrowStruct,
		rightEyebrowStruct,
		mouthStruct,
		UNKNOWN
	};

	return currentFace;
}

vector<Point2i> extractLandMarks(vector<Point2i> landMarks, int index_start, int index_end){
	vector<Point2i> extractedLandmarks;

	for (int i = index_start; i <= index_end; i++)
	{
		extractedLandmarks.push_back(landMarks[i]);
	}

	return extractedLandmarks;
}

vector<Point2i> getLeftEye(vector<Point2i> landMarks){
	vector<Point2i> leftEyeLandMarks = extractLandMarks(landMarks, 36, 41);
	return leftEyeLandMarks;
}

vector<Point2i> getRightEye(vector<Point2i> landMarks){
	vector<Point2i> rightEyeLandMarks = extractLandMarks(landMarks, 42, 47);
	return rightEyeLandMarks;
}

vector<Point2i> getLeftEyebrow(vector<Point2i> landMarks){
	vector<Point2i> leftEyebrowLandMarks = extractLandMarks(landMarks, 17, 21);
	return leftEyebrowLandMarks;
}

vector<Point2i> getRightEyebrow(vector<Point2i> landMarks){
	vector<Point2i> rightEyebrowLandMarks = extractLandMarks(landMarks, 22, 26);
	return rightEyebrowLandMarks;
}

vector<Point2i> getMouth(vector<Point2i> landMarks){
	vector<Point2i> mouthLandMarks = extractLandMarks(landMarks, 48, 67);
	return mouthLandMarks;
}

eye_struct		getFeatureEye(vector<Point2i> eyeLandmarks){
	eye_struct eyeStruct = { getHeightEye(eyeLandmarks) };
	return eyeStruct;
}

eyebrow_struct	getFeatureEyebrow(vector<Point2i> eyebrowLandmarks, vector<Point2i> eyeLandmarks){
	eyebrow_struct eyebrowStruct = { getHeightEyebrow(eyebrowLandmarks), getHeightEyebrow2Eye(eyebrowLandmarks, eyeLandmarks) };
	return eyebrowStruct;
}

mouth_struct	getFeatureMouth(vector<Point2i> mouthLandmarks){
	mouth_struct mouthStruct = { getWidthMouth(mouthLandmarks), getHeightMouth(mouthLandmarks) };
	return mouthStruct;
}

double distanceLineToPoint(vector<Point2i> line, Point2i point){
	if (line.size() != 2) return -1;
	double denom = abs(
		(line[1].y - line[0].y)*point.x -
		(line[1].x - line[0].x)*point.y +
		line[1].x*line[0].y -
		line[1].y*line[0].x);
	double divis = sqrt(pow(line[1].y - line[0].y, 2) + pow(line[1].x - line[0].x, 2));
	return denom / divis;
}

double getHeightEye(vector<Point2i> eyeLandmarks){

	//Obtain midway point from the twin on both eyelids, then find the distance between them.
	//Note that on both eyes, the point 1 2 and 4 5 are the eyelid ones, while the point 0 and 3 are the corners.

	Point2i topEyelid =
	{
		((eyeLandmarks[1].x) + (eyeLandmarks[2].x)) / 2,
		((eyeLandmarks[1].y) + (eyeLandmarks[2].y)) / 2
	};

	Point2i bottomEyelid =
	{
		((eyeLandmarks[4].x) + (eyeLandmarks[5].x)) / 2,
		((eyeLandmarks[4].y) + (eyeLandmarks[5].y)) / 2
	};

	return cv::norm(topEyelid - bottomEyelid);
}

double getHeightEyebrow(vector<Point2i> eyebrowLandmarks)
{
	//Obtain midway point from the start and end of the eyebrow, then find the distance of them to the normally highest point of the eyebrow.
	//Peak would be 2, while edges would be 0 and 4, could be done with line as well.

	Point2i centerEyebrows =
	{
		((eyebrowLandmarks[0].x) + (eyebrowLandmarks[4].x)) / 2,
		((eyebrowLandmarks[0].y) + (eyebrowLandmarks[4].y)) / 2
	};

	return cv::norm(eyebrowLandmarks[2] - centerEyebrows);
}

double getHeightEyebrow2Eye(vector<Point2i> eyeLandmarks, vector<Point2i> eyebrowLandmarks){
	Point2i topEyelid =
	{
		((eyeLandmarks[1].x) + (eyeLandmarks[2].x)) / 2,
		((eyeLandmarks[1].y) + (eyeLandmarks[2].y)) / 2
	};

	return cv::norm(eyebrowLandmarks[2] - topEyelid);
}

double getWidthMouth(vector<Point2i> mouthLandmarks){
	return cv::norm(mouthLandmarks[9] - mouthLandmarks[3]);
}

double getHeightMouth(vector<Point2i> mouthLandmarks){
	return cv::norm(mouthLandmarks[6] - mouthLandmarks[0]);
}

EEmotion getEmotion(face_struct actualFace){
	EEmotion foundEmotion = UNKNOWN;

	#if KNN
		double low = 0;
		vector<pair<double, EEmotion> > distances;
		for (int i = 0; i < facePopulation.size(); i++){
			double distance = getDistance(actualFace, facePopulation[i]);
		
			distances.push_back(make_pair(distance, facePopulation[i].emotion));
		}

		sort(distances.begin(), distances.end());
	
		map<EEmotion, int> neighboursVotes;
	
		for (int i = 0; i <= neighbours && i<facePopulation.size(); i++){
			neighboursVotes[distances[i].second] += 1;
		}
	
		double nrOfVotes = min(neighbours, (int)facePopulation.size());
	
		
		for (map<EEmotion, int>::iterator it = neighboursVotes.begin(); it != neighboursVotes.end(); ++it){
			nrOfVotes += it->second;
			distanceGambiarra[emotionMap[it->first]] = (it->second / nrOfVotes) * 100;
			if (low < it->second){
				low = it->second;
				foundEmotion = it->first;
			}
		}
	#else
		double low = 99999;
		for (int i = 0; i < facePopulation.size(); i++)
		{
			double distance = getDistance(actualFace, facePopulation[i]);

			distanceGambiarra[emotionMap[facePopulation[i].emotion]] = distance;

			if (distance < low)
			{
				low = distance;
				foundEmotion = facePopulation[i].emotion;
			}
		}
	#endif
	return foundEmotion;
}

double getDistance(face_struct actualFace, face_struct baseFace){

	double distance = 0;

	distance += getEyeDistance(actualFace.leftEye, baseFace.leftEye);
	distance += getEyeDistance(actualFace.rightEye, baseFace.rightEye);
	distance += getEyebrowDistance(actualFace.leftEyebrow, baseFace.leftEyebrow);
	distance += getEyebrowDistance(actualFace.rightEyebrow, baseFace.rightEyebrow);
	distance += getMouthDistance(actualFace.mouth, baseFace.mouth);

	return distance;
}

double getEyeDistance(eye_struct actualEye, eye_struct baseEye)
{
	return abs(actualEye.eye_height - baseEye.eye_height);
}

double getEyebrowDistance(eyebrow_struct actualEyebrow, eyebrow_struct baseEyebrow)
{
	return abs(actualEyebrow.eyebrow_height - baseEyebrow.eyebrow_height) + abs(actualEyebrow.eyebrow2eye_height - baseEyebrow.eyebrow2eye_height);
}

double getMouthDistance(mouth_struct actualMouth, mouth_struct baseMouth)
{
	return abs(actualMouth.mouth_height - baseMouth.mouth_height) + abs(actualMouth.mouth_width - baseMouth.mouth_width);
}

vector<double> featureSnapshot(face_struct currentFace){
	vector<double> features;

	double leftEyeDistance = getEyeDistance(currentFace.leftEye, eye_struct{ 0 });
	double rightEyeDistance = getEyeDistance(currentFace.rightEye, eye_struct{ 0 });
	double leftEyebrowDistance = getEyebrowDistance(currentFace.leftEyebrow, eyebrow_struct{ 0 });
	double rightEyebrowDistance = getEyebrowDistance(currentFace.rightEyebrow, eyebrow_struct{ 0 });
	double mouthDistance = getMouthDistance(currentFace.mouth, mouth_struct{ 0 });

	features.push_back(leftEyeDistance);
	features.push_back(rightEyeDistance);
	features.push_back(leftEyebrowDistance);
	features.push_back(rightEyebrowDistance);
	features.push_back(mouthDistance);

	return features;
}

void populateFaces(){
	ifstream facesDatabase;
	facesDatabase.open("learningFaces.csv");
	string input;
	getline(facesDatabase, input);
	while (getline(facesDatabase, input))
	{
		try{

			double leftEyeHeight;
			double rightEyeHeight;
			double leftEyebrowHeight;
			double rightEyebrowHeight;
			double leftEyebrow2eyeHeight;
			double rightEyebrow2eyeHeight;
			double mouthHeight;
			double mouthWidth;
			EEmotion emotion = UNKNOWN;

			stringstream  lineStream(input);
			string        cell;

			std::getline(lineStream, cell, ',');
			for (int i = 0; i < emotionVec.size(); i++){
				if (cell == emotionMap[emotionVec[i]])
					emotion = emotionVec[i];
			}

			std::getline(lineStream, cell, ',');
			leftEyeHeight = stod(cell);
			
			std::getline(lineStream, cell, ',');
			rightEyeHeight = stod(cell);
			
			std::getline(lineStream, cell, ',');
			leftEyebrowHeight = stod(cell);
			
			std::getline(lineStream, cell, ',');
			rightEyebrowHeight = stod(cell);
			
			std::getline(lineStream, cell, ',');
			leftEyebrow2eyeHeight = stod(cell);
			
			std::getline(lineStream, cell, ',');
			rightEyebrow2eyeHeight = stod(cell);
			
			std::getline(lineStream, cell, ',');
			mouthWidth = stod(cell);

			std::getline(lineStream, cell, ',');
			mouthHeight = stod(cell);

			face_struct modelFace =
			{
				eye_struct{ leftEyeHeight },
				eye_struct{ rightEyeHeight },
				eyebrow_struct{ leftEyebrowHeight, leftEyebrow2eyeHeight },
				eyebrow_struct{ rightEyebrowHeight, rightEyebrow2eyeHeight },
				mouth_struct{ mouthWidth, mouthHeight },
				emotion
			};

			facePopulation.push_back(modelFace);
		}
		catch (...){
			facePopulation.clear();
			INFO_STREAM("Error During the Expression Learning Phase. Please Verify the Base and Restart the Algorithm!\n" <<
				"This Session Will Be Started With No Previous Training, Please Calibrate the Algorithm Pressing I!");
			break;
		}
	}
}

void exportFaces(){
	ofstream facesDatabase;
	facesDatabase.open("learningFaces.csv");

	facesDatabase << "Expression,Left Eye Height,Right Eye Height,Left Eyebrow Height,Right Eyebrow Height,Left Eyebrow to Eye Height,Right Eyebrow to Eye Height,Mouth Width,Mouth Height,\n";
	for (int i = 0; i < facePopulation.size(); i++)
	{
		string output =
			emotionMap[facePopulation[i].emotion] + "," +
			to_string(facePopulation[i].leftEye.eye_height) + "," +
			to_string(facePopulation[i].rightEye.eye_height) + "," +
			to_string(facePopulation[i].leftEyebrow.eyebrow_height) + "," +
			to_string(facePopulation[i].rightEyebrow.eyebrow_height) + "," +
			to_string(facePopulation[i].leftEyebrow.eyebrow2eye_height) + "," +
			to_string(facePopulation[i].rightEyebrow.eyebrow2eye_height) + "," +
			to_string(facePopulation[i].mouth.mouth_width) + "," +
			to_string(facePopulation[i].mouth.mouth_height) + ",\n";

		facesDatabase << output;
	};

	facesDatabase.close();
}

void inputFace(face_struct currentFace, EEmotion emotion){
	currentFace.emotion = emotion;
	facePopulation.push_back(currentFace);
}

void pushFaceToFile(face_struct currentFace)
{
	string output =
		emotionMap[currentFace.emotion] + "," +
		to_string(currentFace.leftEye.eye_height) + "," +
		to_string(currentFace.rightEye.eye_height) + "," +
		to_string(currentFace.leftEyebrow.eyebrow_height) + "," +
		to_string(currentFace.rightEyebrow.eyebrow_height) + "," +
		to_string(currentFace.leftEyebrow.eyebrow2eye_height) + "," +
		to_string(currentFace.rightEyebrow.eyebrow2eye_height) + "," +
		to_string(currentFace.mouth.mouth_width) + "," +
		to_string(currentFace.mouth.mouth_height) + "," +
		getSessionTime() + ",\n";

	outputCSV << output;
}

string getSessionName(){
	struct tm *current;
	time_t timenow;
	time(&timenow);
	sessionTime = timenow;
	current = localtime(&timenow);
	string sessionName = "Session-";
	sessionName += to_string(current->tm_year + 1900) + "-";
	sessionName += to_string(current->tm_mon + 1) + "-";
	sessionName += to_string(current->tm_mday) + "-";
	sessionName += to_string(current->tm_hour) + "-";
	sessionName += to_string(current->tm_min) + "-";
	sessionName += to_string(current->tm_sec) + ".csv";
	return sessionName;
}

string getSessionTime()
{
	struct tm *current;
	time_t timenow;
	time(&timenow);
	current = localtime(&timenow);
	double sessionDifference = difftime(timenow, sessionTime);
	string timePassed = to_string((clock() - captureStart) / (double)CLOCKS_PER_SEC) + "ms";
	return timePassed;
}