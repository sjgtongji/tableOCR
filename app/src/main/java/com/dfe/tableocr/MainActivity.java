package com.dfe.tableocr;

import androidx.appcompat.app.AppCompatActivity;

import android.location.LocationManager;
import android.os.Bundle;
import android.util.Log;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import static org.opencv.imgproc.Imgproc.COLOR_GRAY2BGR;

public class MainActivity extends AppCompatActivity {
    public static String test_file_path;
    private static final String TAG = "MainActivity";
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {

        //Connect to OpenCV manager service and initialize
        @Override
        public void onManagerConnected(int status) {
            switch (status){
                case BaseLoaderCallback.SUCCESS:
                    Log.d(TAG, "OpenCV Success");
//                    test();
//                    readTable();
//                    removeBg();
                    new Thread(new Runnable() {
                        @Override
                        public void run() {
                            testHoughLinesP();
                        }
                    }).start();

                    break;
                default:
                    super.onManagerConnected(status);
                    Log.d(TAG, "OpenCV Fail");
                    break;
            }

        }
    };
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
//        Imgcodecs.imread();
//        Imgproc.cvtColor();
//        System.loadLibrary("libopencv_java3.so");
        test_file_path = getExternalCacheDir().getAbsolutePath();
        Log.e("MainActivity" , test_file_path);
//        readTable();
    }

    @Override
    protected void onResume() {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, getApplicationContext(), mLoaderCallback);
        Log.d(TAG, "On Resume OK");
    }

    private void test(){
        long startTime = System.currentTimeMillis();
        Mat src = Imgcodecs.imread( test_file_path + File.separator + "test1.jpg");
        if(src.empty()){
            System.out.println( "not found file" );
            return;
        }
        Mat gray = new Mat();
        Mat erod = new Mat();
        Mat blur = new Mat();
        int src_height = src.cols(), src_width=src.rows();
        //先转为灰度   cvtColor(src, gray, COLOR_BGR2GRAY);
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY);

        /**
         * 腐蚀（黑色区域变大） 
             Mat element = getStructuringElement(MORPH_RECT, Size(erodeSize, erodeSize));
             erode(gray, erod, element);
         */
        int erodeSize = src_height / 200;
        if (erodeSize % 2 == 0){
            erodeSize++;
        }
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(erodeSize, erodeSize));
        Imgproc.erode(gray, erod, element);

        //高斯模糊化
        int blurSize = src_height / 200;
        if (blurSize % 2 == 0) {
            blurSize++;
        }
        Imgproc.GaussianBlur(erod, blur,new Size(blurSize, blurSize), 0, 0);


        //封装的二值化  adaptiveThreshold(~gray, thresh, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 15, -2);
        Mat thresh = gray.clone();
        Mat xx = new Mat();
        Core.bitwise_not(gray,xx);//反色
        Imgproc.adaptiveThreshold(xx, thresh, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, 15, -2);
        imshow(thresh , "thresh.png");
		/*
		这部分的思想是将线条从横纵的方向处理后抽取出来，再进行交叉，矩形的点，进而找到矩形区域的过程

		*/
        // Create the images that will use to extract the horizonta and vertical lines
        //使用二值化后的图像来获取表格横纵的线
        Mat horizontal = thresh.clone();
        Mat vertical = thresh.clone();
        //这个值越大，检测到的直线越多
        int scale = 30; // play with this variable in order to increase/decrease the amount of lines to be detected  使用这个变量来增加/减少待检测的行数


        // Specify size on horizontal axis 指定水平轴上的大小
        int horizontalsize = horizontal.cols() / scale;
        // Create structure element for extracting horizontal lines through morphology operations 创建通过形态学运算提取水平线的结构元素
        // 为了获取横向的表格线，设置腐蚀和膨胀的操作区域为一个比较大的横向直条
        Mat horizontalStructure = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,new Size(horizontalsize, 1));
        // Apply morphology operations
        // 先腐蚀再膨胀
        // iterations 最后一个参数，迭代次数，越多，线越多。在页面清晰的情况下1次即可。
        Imgproc.erode(horizontal, horizontal, horizontalStructure,new Point(-1, -1),1 );
        Imgproc.dilate(horizontal, horizontal, horizontalStructure,new Point(-1, -1),1);
        // dilate(horizontal, horizontal, horizontalStructure, Point(-1, -1)); // expand horizontal lines
        imshow(horizontal , "horizontal.png");
        // Specify size on vertical axis 同上
        int verticalsize = vertical.rows() / scale;
        // Create structure element for extracting vertical lines through morphology operations
        Mat verticalStructure = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,new Size(1, verticalsize));
        Imgproc.erode(vertical, vertical, verticalStructure,new Point(-1, -1),1);
        Imgproc.dilate(vertical, vertical, verticalStructure, new Point(-1, -1),1);
        imshow(vertical , "vertical.png");
		/*
         * 合并线条
         * 将垂直线，水平线合并为一张图
         */
        Mat mask = new Mat();
        Core.add(horizontal,vertical,mask);
        imshow(mask , "mask.png");
	/*
         * 通过 bitwise_and 定位横线、垂直线交汇的点
         */
        Mat joints=new Mat();
        Core.bitwise_and(horizontal, vertical, joints);
        imshow(joints , "joints.png");
	/*
         * 通过 findContours 找轮廓
         *
         * 第一个参数，是输入图像，图像的格式是8位单通道的图像，并且被解析为二值图像（即图中的所有非零像素之间都是相等的）。
         * 第二个参数，是一个 MatOfPoint 数组，在多数实际的操作中即是STL vectors的STL vector，这里将使用找到的轮廓的列表进行填充（即，这将是一个contours的vector,其中contours[i]表示一个特定的轮廓，这样，contours[i][j]将表示contour[i]的一个特定的端点）。
         * 第三个参数，hierarchy，这个参数可以指定，也可以不指定。如果指定的话，输出hierarchy，将会描述输出轮廓树的结构信息。0号元素表示下一个轮廓（同一层级）；1号元素表示前一个轮廓（同一层级）；2号元素表示第一个子轮廓（下一层级）；3号元素表示父轮廓（上一层级）
         * 第四个参数，轮廓的模式，将会告诉OpenCV你想用何种方式来对轮廓进行提取，有四个可选的值：
         *      CV_RETR_EXTERNAL （0）：表示只提取最外面的轮廓；
         *      CV_RETR_LIST （1）：表示提取所有轮廓并将其放入列表；
         *      CV_RETR_CCOMP （2）:表示提取所有轮廓并将组织成一个两层结构，其中顶层轮廓是外部轮廓，第二层轮廓是“洞”的轮廓；
         *      CV_RETR_TREE （3）：表示提取所有轮廓并组织成轮廓嵌套的完整层级结构。
         * 第五个参数，见识方法，即轮廓如何呈现的方法，有三种可选的方法：
         *      CV_CHAIN_APPROX_NONE （1）：将轮廓中的所有点的编码转换成点；
         *      CV_CHAIN_APPROX_SIMPLE （2）：压缩水平、垂直和对角直线段，仅保留它们的端点；
         *      CV_CHAIN_APPROX_TC89_L1  （3）or CV_CHAIN_APPROX_TC89_KCOS（4）：应用Teh-Chin链近似算法中的一种风格
         * 第六个参数，偏移，可选，如果是定，那么返回的轮廓中的所有点均作指定量的偏移
         */
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(mask,contours,hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE,new Point(0,0));

        List<MatOfPoint> contours_poly = contours;
        Rect[] boundRect = new Rect[contours.size()];
        List<Mat> tables = new ArrayList<Mat>();
//my
        List<Rect> haveReacts = new ArrayList();
        Map<String, Map<String, Map<String, Double>>> mappoint=new HashMap<String, Map<String, Map<String, Double>>>();
        //循环所有找到的轮廓-点
        for(int i=0 ; i< contours.size(); i++){
            //每个表的点
            MatOfPoint point = contours.get(i);
            MatOfPoint contours_poly_point = contours_poly.get(i);
/*
             * 获取区域的面积
             * 第一个参数，InputArray contour：输入的点，一般是图像的轮廓点
             * 第二个参数，bool oriented = false:表示某一个方向上轮廓的的面积值，顺时针或者逆时针，一般选择默认false
             */
            double area = Imgproc.contourArea(contours.get(i));
//如果小于某个值就忽略，代表是杂线不是表格
            if(area < 20){continue;}
/*
             * approxPolyDP 函数用来逼近区域成为一个形状，true值表示产生的区域为闭合区域。比如一个带点幅度的曲线，变成折线
             *
             * MatOfPoint2f curve：像素点的数组数据。
             * MatOfPoint2f approxCurve：输出像素点转换后数组数据。
             * double epsilon：判断点到相对应的line segment 的距离的阈值。（距离大于此阈值则舍弃，小于此阈值则保留，epsilon越小，折线的形状越“接近”曲线。）
             * bool closed：曲线是否闭合的标志位。
             */
            Imgproc.approxPolyDP(new MatOfPoint2f(point.toArray()),new MatOfPoint2f(contours_poly_point.toArray()),3,true);
            //为将这片区域转化为矩形，此矩形包含输入的形状
            boundRect[i] = Imgproc.boundingRect(contours_poly.get(i));
// 找到交汇处的的表区域对象
            Mat table_image = joints.submat(boundRect[i]);

            List<MatOfPoint> table_contours = new ArrayList<MatOfPoint>();
            Mat joint_mat = new Mat();
            Imgproc.findContours(table_image, table_contours,joint_mat, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);
//从表格的特性看，如果这片区域的点数小于4，那就代表没有一个完整的表格，忽略掉
            if (table_contours.size() < 4){continue; }

//表格里面的每个点 
            Map<String, Double> x_zhis=new HashMap<String, Double>();
            Map<String, Double> y_zhis=new HashMap<String, Double>();
            for (MatOfPoint matOfPoint : table_contours) {
                Point[] array = matOfPoint.toArray();
                for (Point point2 : array) { x_zhis.put("x"+point2.x, point2.x); y_zhis.put("y"+point2.y, point2.y); }
            }
//System.out.println( boundRect[i].x+"|"+boundRect[i].y+"|"+boundRect[i].width+"|"+boundRect[i].height+"|"+table_contours.size()+">>>>>>>>>>>>>>>>>>>");
            //my add
            haveReacts.add( boundRect[i]);
            Map<String, Map<String, Double>> x =new HashMap<String, Map<String,Double>>(); x.put("x", x_zhis);x.put("y", y_zhis);
            mappoint.put("key"+(haveReacts.size()-1),x);
//保存图片
            tables.add(src.submat(boundRect[i]).clone());
//将矩形画在原图上
            Imgproc.rectangle(src, boundRect[i].tl(), boundRect[i].br(), new Scalar(255, 0, 255), 1, 8, 0);

        }

//页面数据
//        Map<String,String> jspdata=new HashMap<String, String>();
//
//        for(int i=0; i< tables.size(); i++ ){ Mat table = tables.get(i); Rect rect = haveReacts.get(i);
//            int width = rect.width,height=rect.height;
//            Map<String, Map<String, Double>> mapdata = mappoint.get("key"+i);
//            int[] x_z = maptoint(mapdata.get("x"));
//            int[] y_z = maptoint(mapdata.get("y"));
//
//            //纵切 
//            int x_len=0,x_biao=5;
//            List<Mat> mats=new ArrayList<Mat>();
//            for (int j = 0; j < x_z.length; j++) {
//                if(j==0){
//                    Mat img=new Mat(table,new Rect(0,0,x_z[j],height ));if(img.cols()>x_biao ){ mats.add(img); x_len++;}
//                }else{
//                    Mat img=new Mat(table,new Rect(x_z[j-1],0,x_z[j]-x_z[j-1],height )); if(img.cols()>x_biao ){mats.add(img);x_len++;}
//                    if(j == x_z.length - 1){//最后一个处理
//                        Mat img1=new Mat(table,new Rect(x_z[x_z.length - 1],0,width - x_z[x_z.length-1], height ));
//                        if(img.cols()>x_biao ){
//                            mats.add(img1);
//                        }
//                    }
//                }
//            }
//            imshow(table,"table_"+i+".png");//当前table图
//            //横切保存
//            int y_len=0,y_biao=5;
//            for (int j = 0; j <mats.size() ; j++) {Mat mat = mats.get(j);
//                int tuwidth = mat.cols(),tugao = mat.rows();
//                int cy_len=0;
//                for (int k = 0; k < y_z.length; k++) {
//                    if(k==0){
//                        Mat img=new Mat(mat,new Rect(0,0,tuwidth , y_z[k] ));if(img.rows()>y_biao ){
//                            imshow(img,"table_"+i+"_"+j+"_"+cy_len+".png");
//                            cy_len++;
//                        }
//                    }else{
//                        Mat img=new Mat(mat,new Rect(0,y_z[k-1],tuwidth,y_z[k]-y_z[k-1]));if(img.rows()>y_biao ){
//                            imshow(img,"table_"+i+"_"+j+"_"+cy_len+".png");cy_len++;
//                        }
//                        if(k==y_z.length-1){//最后一个处理  
//                            Mat img1=new Mat(mat,new Rect(0,y_z[k],tuwidth,tugao-y_z[k] ));if(img.rows()>y_biao ){
//                                imshow(img1,"table_"+i+"_"+j+"_"+(cy_len)+".png");
//                            }
//                        }
//                    }
//                }
//                y_len=cy_len;
//            }
//            //保存数据信息
//            jspdata.put("table_"+i, x_len+"_"+y_len);
//        }
        long endTime=System.currentTimeMillis();
    }
//
//    private int[] maptoint(Map<String,Double> x) {
//        return null;
//    }
//
    private void imshow(Mat img, String s) {
        String outPath = this.test_file_path + File.separator + s;

        File file = new File(outPath);
        //目录是否存在
        if(!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        Imgcodecs.imwrite(outPath,img);
    }


    private void test1(){
        File file = new File(test_file_path + File.separator + "test.txt");
        if(!file.exists()){
            try {
                file.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
    public void removeBg(){
        Mat source_image = Imgcodecs.imread(test_file_path + "/test1.jpg");
        Mat gray_image = new Mat(source_image.height(), source_image.width(), CvType.CV_8UC1);
        Imgproc.cvtColor(source_image,gray_image,Imgproc.COLOR_RGB2GRAY);
        Mat thresh_image = new Mat(source_image.height(), source_image.width(), CvType.CV_8UC1);
        int th = 210;
        Imgproc.threshold(gray_image, thresh_image, 230, 255, Imgproc.THRESH_BINARY);
        this.saveImage("thresh_image.png", thresh_image);
        List<Mat> mv = new ArrayList<Mat>();
        Core.split(source_image, mv);
//        for(int i = 0 ; i < mv.size() ; i++){
//            this.saveImage("split-" + i + ".png",mv.get(i));
//        }
        Mat red = mv.get(2);
        Mat blue = mv.get(0);
        Mat green = mv.get(1);

        Mat red_binary = new Mat();
        Mat blue_binary = new Mat();
        Mat green_binary = new Mat();

        Imgproc.threshold(red, red_binary, 230, 255, Imgproc.THRESH_BINARY);
        Imgproc.threshold(blue, blue_binary, 250, 255, Imgproc.THRESH_BINARY);
        Imgproc.threshold(green, green_binary, 200, 255, Imgproc.THRESH_BINARY);
        this.saveImage("red.png" , red_binary);
        this.saveImage("blue.png" , blue_binary);
        this.saveImage("green.png" , green_binary);
        Mat result = new Mat();
        Core.bitwise_not(red_binary , result);
        this.saveImage("result.png" , result);
    }
    /**
     * 读取 table
     */
    public void readTable(){
        Mat source_image = Imgcodecs.imread(test_file_path + "/test4.jpg");
        List<Mat> mv = new ArrayList<Mat>();
        Core.split(source_image, mv);
//        for(int i = 0 ; i < mv.size() ; i++){
//            this.saveImage("split-" + i + ".png",mv.get(i));
//        }
        Mat red = mv.get(2);

        Mat red_binary = new Mat();


        Imgproc.threshold(red, red_binary, 230, 255, Imgproc.THRESH_BINARY);

        this.saveImage("red.png" , red_binary);

        Mat thresh_image = new Mat();
        Core.bitwise_not(red_binary , thresh_image);
//        //灰度处理
//        Mat gray_image = new Mat(source_image.height(), source_image.width(), CvType.CV_8UC1);
//        Imgproc.cvtColor(source_image,gray_image,Imgproc.COLOR_RGB2GRAY);

//        //二值化
//        Mat thresh_image = new Mat(source_image.height(), source_image.width(), CvType.CV_8UC1);
//
//        // C 负数，取反色，超过阈值的为黑色，其他为白色
//        Imgproc.adaptiveThreshold(gray_image , thresh_image,255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY,7,-2);
        this.saveImage("1-thresh.png",thresh_image);
//        if(true){
//            return;
//        }


        //克隆一个 Mat，用于提取水平线
        Mat horizontal_image = thresh_image.clone();

        //克隆一个 Mat，用于提取垂直线
        Mat vertical_image = thresh_image.clone();

        /*
         * 求水平线
         * 1. 根据页面的列数（可以理解为宽度），将页面化成若干的扫描区域
         * 2. 根据扫描区域的宽度，创建一根水平线
         * 3. 通过腐蚀、膨胀，将满足条件的区域，用水平线勾画出来
         *
         * scale 越大，识别的线越多，因为，越大，页面划定的区域越小，在腐蚀后，多行文字会形成一个块，那么就会有一条线
         * 在识别表格时，我们可以理解线是从页面左边 到 页面右边的，那么划定的区域越小，满足的条件越少，线条也更准确
         */
        int scale = 30;
        int horizontalsize = horizontal_image.cols() / scale;
        // 为了获取横向的表格线，设置腐蚀和膨胀的操作区域为一个比较大的横向直条
        Mat horizontalStructure = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(horizontalsize, 1));
        // 先腐蚀再膨胀 new Point(-1, -1) 以中心原点开始
        // iterations 最后一个参数，迭代次数，越多，线越多。在页面清晰的情况下1次即可。
        Imgproc.erode(horizontal_image, horizontal_image, horizontalStructure, new Point(-1, -1),1);
        Imgproc.dilate(horizontal_image, horizontal_image, horizontalStructure, new Point(-1, -1),1);
        this.saveImage("2-horizontal.png",horizontal_image);

        // 求垂直线
        scale = 60;
        int verticalsize = vertical_image.rows() / scale;
        Mat verticalStructure = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(1, verticalsize));
        Imgproc.erode(vertical_image, vertical_image, verticalStructure, new Point(-1, -1),1);
        Imgproc.dilate(vertical_image, vertical_image, verticalStructure, new Point(-1, -1),1);
        this.saveImage("3-vertical.png",vertical_image);

        /*
         * 合并线条
         * 将垂直线，水平线合并为一张图
         */
        Mat mask_image = new Mat();
        Core.add(horizontal_image,vertical_image,mask_image);
        this.saveImage("4-mask.png",mask_image);

        /*
         * 通过 bitwise_and 定位横线、垂直线交汇的点
         */
        Mat points_image = new Mat();
        Core.bitwise_and(horizontal_image, vertical_image, points_image);
        this.saveImage("5-points.png",points_image);

        /*
         * 通过 findContours 找轮廓
         *
         * 第一个参数，是输入图像，图像的格式是8位单通道的图像，并且被解析为二值图像（即图中的所有非零像素之间都是相等的）。
         * 第二个参数，是一个 MatOfPoint 数组，在多数实际的操作中即是STL vectors的STL vector，这里将使用找到的轮廓的列表进行填充（即，这将是一个contours的vector,其中contours[i]表示一个特定的轮廓，这样，contours[i][j]将表示contour[i]的一个特定的端点）。
         * 第三个参数，hierarchy，这个参数可以指定，也可以不指定。如果指定的话，输出hierarchy，将会描述输出轮廓树的结构信息。0号元素表示下一个轮廓（同一层级）；1号元素表示前一个轮廓（同一层级）；2号元素表示第一个子轮廓（下一层级）；3号元素表示父轮廓（上一层级）
         * 第四个参数，轮廓的模式，将会告诉OpenCV你想用何种方式来对轮廓进行提取，有四个可选的值：
         *      CV_RETR_EXTERNAL （0）：表示只提取最外面的轮廓；
         *      CV_RETR_LIST （1）：表示提取所有轮廓并将其放入列表；
         *      CV_RETR_CCOMP （2）:表示提取所有轮廓并将组织成一个两层结构，其中顶层轮廓是外部轮廓，第二层轮廓是“洞”的轮廓；
         *      CV_RETR_TREE （3）：表示提取所有轮廓并组织成轮廓嵌套的完整层级结构。
         * 第五个参数，见识方法，即轮廓如何呈现的方法，有三种可选的方法：
         *      CV_CHAIN_APPROX_NONE （1）：将轮廓中的所有点的编码转换成点；
         *      CV_CHAIN_APPROX_SIMPLE （2）：压缩水平、垂直和对角直线段，仅保留它们的端点；
         *      CV_CHAIN_APPROX_TC89_L1  （3）or CV_CHAIN_APPROX_TC89_KCOS（4）：应用Teh-Chin链近似算法中的一种风格
         * 第六个参数，偏移，可选，如果是定，那么返回的轮廓中的所有点均作指定量的偏移
         */
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(mask_image,contours,hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE,new Point(0,0));


        List<MatOfPoint> contours_poly = contours;
        Rect[] boundRect = new Rect[contours.size()];

        LinkedList<Mat> tables = new LinkedList<Mat>();

        //循环所有找到的轮廓-点
        for(int i=0 ; i< contours.size(); i++){

            MatOfPoint point = contours.get(i);
            MatOfPoint contours_poly_point = contours_poly.get(i);

            /*
             * 获取区域的面积
             * 第一个参数，InputArray contour：输入的点，一般是图像的轮廓点
             * 第二个参数，bool oriented = false:表示某一个方向上轮廓的的面积值，顺时针或者逆时针，一般选择默认false
             */
            double area = Imgproc.contourArea(contours.get(i));
            //如果小于某个值就忽略，代表是杂线不是表格
            if(area < 100){
                continue;
            }

            /*
             * approxPolyDP 函数用来逼近区域成为一个形状，true值表示产生的区域为闭合区域。比如一个带点幅度的曲线，变成折线
             *
             * MatOfPoint2f curve：像素点的数组数据。
             * MatOfPoint2f approxCurve：输出像素点转换后数组数据。
             * double epsilon：判断点到相对应的line segment 的距离的阈值。（距离大于此阈值则舍弃，小于此阈值则保留，epsilon越小，折线的形状越“接近”曲线。）
             * bool closed：曲线是否闭合的标志位。
             */
            Imgproc.approxPolyDP(new MatOfPoint2f(point.toArray()),new MatOfPoint2f(contours_poly_point.toArray()),3,true);

            //为将这片区域转化为矩形，此矩形包含输入的形状
            boundRect[i] = Imgproc.boundingRect(contours_poly.get(i));

            // 找到交汇处的的表区域对象
            Mat table_image = points_image.submat(boundRect[i]);

            List<MatOfPoint> table_contours = new ArrayList<MatOfPoint>();
            Mat joint_mat = new Mat();
            Imgproc.findContours(table_image, table_contours,joint_mat, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
            //从表格的特性看，如果这片区域的点数小于4，那就代表没有一个完整的表格，忽略掉
            if (table_contours.size() < 4)
                continue;

            //保存图片
            tables.addFirst(source_image.submat(boundRect[i]).clone());

            //将矩形画在原图上
            Imgproc.rectangle(source_image, boundRect[i].tl(), boundRect[i].br(), new Scalar(0, 255, 0), 1, 8, 0);

        }

        for(int i=0; i< tables.size(); i++ ){

            //拿到表格后，可以对表格再次处理，比如 OCR 识别等
            this.saveImage("6-table-"+(i+1)+".png",tables.get(i));
        }

        this.saveImage("7-source.png",source_image);

    }

    private void testHoughLinesP(){
        Mat source_image = Imgcodecs.imread(test_file_path + "/test4.jpg");
        List<Mat> mv = new ArrayList<Mat>();
        Core.split(source_image, mv);
//        for(int i = 0 ; i < mv.size() ; i++){
//            this.saveImage("split-" + i + ".png",mv.get(i));
//        }
        Mat red = mv.get(2);

        Mat red_binary = new Mat();


        Imgproc.threshold(red, red_binary, 230, 255, Imgproc.THRESH_BINARY);

        this.saveImage("red.png" , red_binary);

        Mat thresh_image = new Mat();
        Core.bitwise_not(red_binary , thresh_image);
        this.saveImage("1-thresh.png",thresh_image);
        Mat linesHorizontal = new Mat();
        Mat linesVertical = new Mat();
//克隆一个 Mat，用于提取水平线
        Mat horizontal_image = thresh_image.clone();

        //克隆一个 Mat，用于提取垂直线
        Mat vertical_image = thresh_image.clone();

        /*
         * 求水平线
         * 1. 根据页面的列数（可以理解为宽度），将页面化成若干的扫描区域
         * 2. 根据扫描区域的宽度，创建一根水平线
         * 3. 通过腐蚀、膨胀，将满足条件的区域，用水平线勾画出来
         *
         * scale 越大，识别的线越多，因为，越大，页面划定的区域越小，在腐蚀后，多行文字会形成一个块，那么就会有一条线
         * 在识别表格时，我们可以理解线是从页面左边 到 页面右边的，那么划定的区域越小，满足的条件越少，线条也更准确
         */
        int scale = 30;
        int horizontalsize = horizontal_image.cols() / scale;
        // 为了获取横向的表格线，设置腐蚀和膨胀的操作区域为一个比较大的横向直条
        Mat horizontalStructure = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(horizontalsize, 1));
        // 先腐蚀再膨胀 new Point(-1, -1) 以中心原点开始
        // iterations 最后一个参数，迭代次数，越多，线越多。在页面清晰的情况下1次即可。
        Imgproc.erode(horizontal_image, horizontal_image, horizontalStructure, new Point(-1, -1),1);
        Imgproc.dilate(horizontal_image, horizontal_image, horizontalStructure, new Point(-1, -1),1);
        this.saveImage("2-horizontal.png",horizontal_image);

        // 求垂直线
        scale = 60;
        int verticalsize = vertical_image.rows() / scale;
        Mat verticalStructure = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(1, verticalsize));
        Imgproc.erode(vertical_image, vertical_image, verticalStructure, new Point(-1, -1),1);
        Imgproc.dilate(vertical_image, vertical_image, verticalStructure, new Point(-1, -1),1);
        this.saveImage("3-vertical.png",vertical_image);

        Imgproc.HoughLinesP(horizontal_image , linesHorizontal , 1, Math.PI / 180, 4 , horizontal_image.width() / 2 , horizontal_image.width() / 20);
//        ArrayList<Double> lineYs = new ArrayList<>();
        ArrayList<Line> lineYs = new ArrayList<>();
//        ArrayList<Double> uniqueLineYs = new ArrayList<>();
        for (int i = 0; i < linesHorizontal.rows(); i++) {
            double[] points = linesHorizontal.get(i, 0);

            // if it slopes, get the average of them, store the y-coordinate
//            if (Math.abs(y1 - y2) < 10) {
//                lineYs.add((y1 + y2) / 2);
//            }
            lineYs.add(point2Line(points));
        }
//        getUniqueLines(lineYs, uniqueLineYs, 10);
        showMarkedLines(thresh_image, lineYs , true);


        Imgproc.HoughLinesP(vertical_image , linesVertical , 1, Math.PI / 180, 1 , 35 , 20);
        ArrayList<Line> lineXs = new ArrayList<>();
//        ArrayList<Double> uniqueLineYs = new ArrayList<>();
        for (int i = 0; i < linesVertical.rows(); i++) {
            double[] points = linesVertical.get(i, 0);


            // if it slopes, get the average of them, store the y-coordinate
//            if (Math.abs(y1 - y2) < 10) {
//                lineYs.add((y1 + y2) / 2);
//            }
            lineXs.add(point2Line(points));
        }
//        getUniqueLines(lineYs, uniqueLineYs, 10);
        showMarkedLines(thresh_image, lineXs , false);
        Log.d(TAG , "testHoughLinesP finished");
    }

    /**
     * filter the source coordinates, if some values are too close ,get the average of them
     *
     * @param src    source coordinates list
     * @param dst    destination coordinate list
     * @param minGap the minimum gap between coordinates
     */
    private void getUniqueLines(ArrayList<Double> src, ArrayList<Double> dst, int minGap) {
        Collections.sort(src); //sort the source coordinates list
        for (int i = 0; i < src.size(); i++) {
            double sum = src.get(i);
            double num = 1;
            //when the distance between lines less than minGap, get the average of thema
            while (i != src.size() - 1 && src.get(i + 1) - src.get(i) < minGap) {
                num++;
                sum = sum + src.get(i + 1);
                i++;
            }
            if (num == 1) {
                dst.add(src.get(i));
            } else {
                dst.add(((sum / num)));
            }
        }
    }
    private void showMarkedLines(Mat src, ArrayList<Line> lines, boolean isHorizontal) {
        Mat showLines = new Mat();
        Imgproc.cvtColor(src, showLines, COLOR_GRAY2BGR);
        if(isHorizontal){
            for (Line line : lines) {

                Point pt1 = new Point(line.x1, line.y1);
                Point pt2 = new Point(line.x2, line.y1);
                Imgproc.line(showLines, pt1, pt2, new Scalar(0, 0, 255), 2);
            }
            saveImage("horizontal_lines.jpg" , showLines);
        }else{
            for(Line line : lines){
                Point pt1 = new Point(line.x1, line.y1);
                Point pt2 = new Point(line.x2, line.y2);
                Imgproc.line(showLines, pt1, pt2, new Scalar(0, 0, 255), 2);
            }
            saveImage("vertical_lines.jpg" , showLines);
        }

    }

    private void saveImage(String path,Mat image){

        String outPath = this.test_file_path + File.separator + path;

        File file = new File(outPath);
        //目录是否存在
        if(!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        Imgcodecs.imwrite(outPath,image);

    }
    public static Line point2Line(double[] point){
        Line line = new Line();
        line.x1 = point[0];
        line.y1 = point[1];
        line.x2 = point[2];
        line.y2 = point[3];
        return line;
    }
    public static class Line{
        public double x1;
        public double y1;
        public double x2;
        public double y2;

    }
}
