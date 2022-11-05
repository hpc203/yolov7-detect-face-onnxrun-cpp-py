#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>

using namespace std;
using namespace cv;
using namespace Ort;

struct Net_config
{
	float confThreshold; // Confidence threshold
	float nmsThreshold;  // Non-maximum suppression threshold
	string modelpath;
};

typedef struct PointInfo
{
	Point pt;
	float score;
} PointInfo;

typedef struct BoxInfo
{
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	PointInfo kpt1;
	PointInfo kpt2;
	PointInfo kpt3;
	PointInfo kpt4;
	PointInfo kpt5;
} BoxInfo;

class YOLOV7_face
{
public:
	YOLOV7_face(Net_config config);
	void detect(Mat& frame);
private:
	int inpWidth;
	int inpHeight;
	int nout;
	int num_proposal;

	float confThreshold;
	float nmsThreshold;
	vector<float> input_image_;
	void normalize_(Mat img);
	void nms(vector<BoxInfo>& input_boxes);
	bool has_postprocess;

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "YOLOV7_face");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> input_node_dims; // >=1 outputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
};

YOLOV7_face::YOLOV7_face(Net_config config)
{
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;

	string model_path = config.modelpath;
	std::wstring widestr = std::wstring(model_path.begin(), model_path.end());
	//OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Session(env, widestr.c_str(), sessionOptions);
	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
	this->inpHeight = input_node_dims[0][2];
	this->inpWidth = input_node_dims[0][3];

}

void YOLOV7_face::normalize_(Mat img)
{
	//    img.convertTo(img, CV_32F);
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(row * col * img.channels());
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				float pix = img.ptr<uchar>(i)[j * 3 + 2 - c];
				this->input_image_[c * row * col + i * col + j] = pix / 255.0;
			}
		}
	}
}

void YOLOV7_face::nms(vector<BoxInfo>& input_boxes)
{
	sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
	vector<float> vArea(input_boxes.size());
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
			* (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
	}

	vector<bool> isSuppressed(input_boxes.size(), false);
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		if (isSuppressed[i]) { continue; }
		for (int j = i + 1; j < int(input_boxes.size()); ++j)
		{
			if (isSuppressed[j]) { continue; }
			float xx1 = (max)(input_boxes[i].x1, input_boxes[j].x1);
			float yy1 = (max)(input_boxes[i].y1, input_boxes[j].y1);
			float xx2 = (min)(input_boxes[i].x2, input_boxes[j].x2);
			float yy2 = (min)(input_boxes[i].y2, input_boxes[j].y2);

			float w = (max)(float(0), xx2 - xx1 + 1);
			float h = (max)(float(0), yy2 - yy1 + 1);
			float inter = w * h;
			float ovr = inter / (vArea[i] + vArea[j] - inter);

			if (ovr >= this->nmsThreshold)
			{
				isSuppressed[j] = true;
			}
		}
	}
	// return post_nms;
	int idx_t = 0;
	input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &isSuppressed](const BoxInfo& f) { return isSuppressed[idx_t++]; }), input_boxes.end());
}

void YOLOV7_face::detect(Mat& frame)
{
	Mat dstimg;
	resize(frame, dstimg, Size(this->inpWidth, this->inpHeight));
	this->normalize_(dstimg);
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	// 开始推理
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());   // 开始推理
	vector<BoxInfo> generate_boxes;
	
	Ort::Value &predictions = ort_outputs.at(0);
	auto pred_dims = predictions.GetTensorTypeAndShapeInfo().GetShape();
	num_proposal = pred_dims.at(1);
	nout = pred_dims.at(2);

	float ratioh = (float)frame.rows / this->inpHeight, ratiow = (float)frame.cols / this->inpWidth;
	int n = 0, k = 0; ///cx,cy,w,h,box_score, class_score, x1,y1,score1, ...., x5,y5,score5
	const float* pdata = predictions.GetTensorMutableData<float>();
	for (n = 0; n < this->num_proposal; n++)   ///特征图尺度
	{
		float box_score = pdata[4];
		if (box_score > this->confThreshold)
		{
			float class_socre = box_score * pdata[5];
			if (class_socre > this->confThreshold)
			{
				float cx = pdata[0] * ratiow;  ///cx
				float cy = pdata[1] * ratioh;   ///cy
				float w = pdata[2] * ratiow;   ///w
				float h = pdata[3] * ratioh;  ///h

				float xmin = cx - 0.5 * w;
				float ymin = cy - 0.5 * h;
				float xmax = cx + 0.5 * w;
				float ymax = cy + 0.5 * h;

				k = 0;
				int x = int(pdata[6 + k] * ratiow);
				int y = int(pdata[6 + k + 1] * ratioh);
				float score = pdata[6 + k + 2];
				PointInfo kpt1 = { Point(x,y), score };
				k += 3;

				x = int(pdata[6 + k] * ratiow);
				y = int(pdata[6 + k + 1] * ratioh);
				score = pdata[6 + k + 2];
				PointInfo kpt2 = { Point(x,y), score };
				k += 3;

				x = int(pdata[6 + k] * ratiow);
				y = int(pdata[6 + k + 1] * ratioh);
				score = pdata[6 + k + 2];
				PointInfo kpt3 = { Point(x,y), score };
				k += 3;

				x = int(pdata[6 + k] * ratiow);
				y = int(pdata[6 + k + 1] * ratioh);
				score = pdata[6 + k + 2];
				PointInfo kpt4 = { Point(x,y), score };
				k += 3;

				x = int(pdata[6 + k] * ratiow);
				y = int(pdata[6 + k + 1] * ratioh);
				score = pdata[6 + k + 2];
				PointInfo kpt5 = { Point(x,y), score };

				generate_boxes.push_back(BoxInfo{ xmin, ymin, xmax, ymax, class_socre, kpt1,kpt2,kpt3,kpt4,kpt5 });
			}
		}
		pdata += nout;
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	nms(generate_boxes);

	for (size_t n = 0; n < generate_boxes.size(); n++)
	{
		int xmin = int(generate_boxes[n].x1);
		int ymin = int(generate_boxes[n].y1);
		rectangle(frame, Point(xmin, ymin), Point(int(generate_boxes[n].x2), int(generate_boxes[n].y2)), Scalar(0, 0, 255), 2);
		string label = format("%.2f", generate_boxes[n].score);
		label = "face:" + label;
		putText(frame, label, Point(xmin, ymin - 5), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
		circle(frame, generate_boxes[n].kpt1.pt, 1, Scalar(0, 255, 0), -1);
		circle(frame, generate_boxes[n].kpt2.pt, 1, Scalar(0, 255, 0), -1);
		circle(frame, generate_boxes[n].kpt3.pt, 1, Scalar(0, 255, 0), -1);
		circle(frame, generate_boxes[n].kpt4.pt, 1, Scalar(0, 255, 0), -1);
		circle(frame, generate_boxes[n].kpt5.pt, 1, Scalar(0, 255, 0), -1);
	}
}

int main()
{
	Net_config YOLOV7_face_cfg = { 0.45, 0.5, "onnx_havepost_models/yolov7-lite-e.onnx" };
	YOLOV7_face net(YOLOV7_face_cfg);
	string imgpath = "test.jpg";
	Mat srcimg = imread(imgpath);
	net.detect(srcimg);

	static const string kWinName = "Deep learning object detection in ONNXRuntime";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, srcimg);
	waitKey(0);
	destroyAllWindows();
}