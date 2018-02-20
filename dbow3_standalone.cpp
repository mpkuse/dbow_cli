/*
    DBOW3 - Example

    1. Read Image
    2. Extract ORB keypoints and descriptor
    3. Add to DB
    4. Query from DB using current
    5. go to step 1

    Author  : Manohar Kuse <mpkuse@connect.ust.hk>
    Created : 20th Feb, 2018
*/


#include <iostream>
#include <iomanip>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

#include "dbow3_src/DBoW3.h"
using namespace DBoW3;

void get_image_fname( const string BASE , int i, string& fname )
{
  // sprintf( *fname, "%s/%04d.jpg", BASE, 2*i+1 );
  char tmp[20];
  sprintf( tmp, "%04d", 2*i+1 );
  fname = BASE + tmp + ".jpg";
}


  // counts number of positives in status. This status is from findFundamentalMat()
  int count_inliers( const vector<uchar>& status )
  {
    int cc = 0;
    for( int i=0 ; i<status.size() ; i++ )
      if( status[i] > 0 )
        cc++;

    return cc;
  }




// Given 2 images with their matches points. ie. pts_1[i] <---> pts_2[i].
// This function returns the plotted with/without lines numbers
void drawMatches( const cv::Mat& im1, const vector<cv::Point2f>& pts_1,
                  const cv::Mat& im2, const vector<cv::Point2f>& pts_2,
                  cv::Mat& outImage,
                  const string msg,
                  vector<uchar> status = vector<uchar>(),
                  bool enable_lines=true, bool enable_points=true)
{
  assert( pts_1.size() == pts_2.size() );
  assert( im1.rows == im2.rows );
  assert( im2.cols == im2.cols );
  assert( im1.channels() == im2.channels() );

  cv::Mat row_im;
  cv::hconcat(im1, im2, row_im);

  if( row_im.channels() == 3 )
    outImage = row_im.clone();
  else
    cvtColor(row_im, outImage, CV_GRAY2RGB);


    // loop  over points
  for( int i=0 ; i<pts_1.size() ; i++ )
  {
    cv::Point2f p1 = pts_1[i];
    cv::Point2f p2 = pts_2[i];

    cv::circle( outImage, p1, 4, cv::Scalar(0,0,255), -1 );
    cv::circle( outImage, p2+cv::Point2f(im1.cols,0), 4, cv::Scalar(0,0,255), -1 );

    cv::line( outImage,  p1, p2+cv::Point2f(im1.cols,0), cv::Scalar(255,0,0) );

    if( status.size() > 0 && status[i] > 0 )
      cv::line( outImage,  p1, p2+cv::Point2f(im1.cols,0), cv::Scalar(0,255,0) );


  }


  if( msg.length() > 0 ) {
    cv::putText( outImage, msg, cv::Point(5,50), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0,0,255) );
  }

}




// Simple geometry verification. Given 2 images compute keypoints,
  // compute descriptors, match descriptors with FLANN and
  // return number of matched features
  int geometric_verify_naive( const cv::Mat& im__1, const std::vector<cv::KeyPoint>& keypts_1, cv::Mat& desc_1,
                              const cv::Mat& im__2, const std::vector<cv::KeyPoint>& keypts_2, cv::Mat& desc_2,
                              cv::Mat& outImg_small )
  {
    cout << "---geometric_verify_naive---\n";

    // detectAndCompute
    // std::vector<cv::KeyPoint> keypts_1, keypts_2;
    // cv::Mat desc_1, desc_2;
    // fdetector->detectAndCompute( im__1, cv::Mat(), keypts_1, desc_1 );
    // fdetector->detectAndCompute( im__2, cv::Mat(), keypts_2, desc_2 );
    cout << "# of keypoints: "<< keypts_1.size() << "  " << keypts_2.size() << endl;
    cout << "desc.shape    : ("<< desc_1.rows << "," << desc_1.cols << ")   (" << desc_2.rows << "," << desc_2.cols << ")\n" ;

    // FLANN index create and match
    if( desc_1.type() != CV_32F )
    {
      desc_1.convertTo( desc_1, CV_32F );
    }
    if( desc_2.type() != CV_32F )
    {
      desc_2.convertTo( desc_2, CV_32F );
    }
    cv::FlannBasedMatcher matcher;
    std::vector< std::vector< cv::DMatch > > matches_raw;
    matcher.knnMatch( desc_1, desc_2, matches_raw, 2 );
    cout << "# Matches : " << matches_raw.size() << endl;
    cout << "# Matches[0] : " << matches_raw[0].size() << endl;



    // ratio test
    vector<cv::DMatch> matches;
    vector<cv::Point2f> pts_1, pts_2;
    for( int j=0 ; j<matches_raw.size() ; j++ )
    {
      if( matches_raw[j][0].distance < 0.8 * matches_raw[j][1].distance ) //good match
      {
        matches.push_back( matches_raw[j][0] );

        // Get points
        int t = matches_raw[j][0].trainIdx;
        int q = matches_raw[j][0].queryIdx;
        pts_1.push_back( keypts_1[q].pt );
        pts_2.push_back( keypts_2[t].pt );
      }
    }
    cout << "# Retained (after ratio test): "<< matches.size() << endl;




    // f-test. You might need to undistort point sets. But if precision is not needed, probably skip it.
    vector<uchar> status;
    cv::findFundamentalMat(pts_1, pts_2, cv::FM_RANSAC, 5.0, 0.99, status);
    int n_inliers = count_inliers( status );
    cout << "# Retained (after fundamental-matrix-test): " << n_inliers << endl;

    // try drawing the matches. Just need to verify.
    // draw
    cv::Mat outImg;
    drawMatches( im__1, pts_1, im__2, pts_2, outImg, to_string( matches.size() )+  " :: "+ to_string( n_inliers ), status );
    cv::resize( outImg, outImg_small, cv::Size(), 0.5, 0.5  );
    // cv::imshow( "outImg", outImg_small );
    // cv::waitKey(10);

    cout << "----------\n";
    return n_inliers;


  }





int main()
{

  // Init - ORB detector
  cout << "INIT - ORB detector" << endl;
  cv::Ptr<cv::Feature2D> fdetector;
  fdetector=cv::ORB::create();


  // Init - DBOW Vocab
  cout << "INIT - DBOW-ORB Vocabulary";
  cout << "Read : " << "../orbvoc.dbow3" << endl;
  Vocabulary voc("../orbvoc.dbow3");
  Database db(voc, false, 0);


  // Init - Image and Attribute Storage
  vector< cv::Mat > vec_im;
  vector< vector<cv::KeyPoint> > vec_keypts;
  vector< cv::Mat > vec_descriptor;


  // Loop over images
  for( int i=0 ; i<1000 ; i++ )
  {
    cout << "---\n";
    // Step-1 : Load Image
    string fname;
    // sprintf( fname, "../images/%04d.jpg", i );
    get_image_fname( "../images/" , i, fname  );
    cout << "Reading Image: " << fname << endl;
    cv::Mat im = cv::imread( fname.c_str() );
    // cv::Mat im = cv::imread( "ddd.jpg" );
    // cv::imshow( "win", im );
    // cv::waitKey(0);


    // Step-2 : Extract Keypoints and Descriptors (ORB)
    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    fdetector->detectAndCompute(im, cv::Mat(), keypoints, descriptors);
    // cout << "# of keypoints : "<< keypoints.size() << endl;
    // cout << "descriptors shape : "<< descriptors.rows << "x" << descriptors.cols << endl;
    // cout<<"done detecting features"<<endl;


    // Step-3 : Add Features descriptor to DB
    BowVector bow_vec;
    db.add( descriptors.clone(), &bow_vec );

    // Step-3.1 : Storage of raw image, keypoints, descriptors (for geometric verification)
    vec_im.push_back( im );
    vec_keypts.push_back( keypoints );
    vec_descriptor.push_back( descriptors );



    // Step-4 : Query current descriptors from the db
    QueryResults ret;
    db.query( descriptors, ret, 20 );
    cout << "Searching for Image " << i << endl;// << ". " << ret << endl;

    for( int j=1 ; j < ret.size() ; j++ )
    {
      if( ret[j].Score > 0.055 ) //0.055
      {
        cout << "Potential Loop : " << i << " <---> " << ret[j].Id << endl;
        // cv::imshow( "1", vec_im[i] );
        // cv::imshow( "2", vec_im[ ret[j].Id ]);

        // Geometric Verification of the loop
        cv::Mat dump;

        int n_inliers = geometric_verify_naive( vec_im[i], vec_keypts[i], vec_descriptor[i],
                                vec_im[ ret[j].Id], vec_keypts[ ret[j].Id], vec_descriptor[ ret[j].Id],
                                dump );
        cout << "n_inliers : " << n_inliers << endl;
        // cv::imshow( "dump", dump );
        // cv::waitKey( 0 );
      }
    }

  }


  cout << "Final Database information: " << endl << db << endl;


}
