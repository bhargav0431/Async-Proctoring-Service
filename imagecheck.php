<?php
require_once("dbconfig.php");
require_once("constants.inc");
//ini_set('display_errors', 1); ini_set('display_startup_errors', 1); error_reporting(E_ALL);
function getMacAddress() {
	system('wmic bios get serialnumber');
    $mycom=ob_get_contents();
    ob_clean();
    $data = trim(str_replace('SerialNumber', '', $mycom)); 
	return $data;
}
if(isset($_GET['page'])){
	$page=$_GET['page'];
}else{
	$page='test';
}

$dataval = json_decode(file_get_contents('php://input'), true);
$membership_no = $dataval['membership_no'];

$getphotosign = "SELECT photo, signature FROM iib_candidate WHERE membership_no='$membership_no'";
$resultgetphotosign = mysql_query($getphotosign, $SLAVECONN) or errorlog('err0143', " QUERY: $getphotosign  " . mysql_error($SLAVECONN));
list($photo_path,$sign_path) = mysql_fetch_row($resultgetphotosign);
$p1 = $canPhotoPath . $photo_path;
/* $sp1 = $canSignPath . $sign_path;	
$p1 = $canPhotoPath . 'P' . $membership_no . '.jpg';
$p2 = $canPhotoPath . 'p' . $membership_no . '.jpg'; */
$memPhoto = './themes/default/images/photo_no.jpg';

if (file_exists($p1))
    $memPhoto = $p1;
else {
    $memPhoto = './themes/default/images/photo_no.jpg';
}

/*$type = pathinfo($memPhoto, PATHINFO_EXTENSION);
$contents = file_get_contents($memPhoto);
$imgdata = 'data:image/' . $type . ';base64,'. base64_encode($contents);
$arrayimg = array("img1"=>$imgdata,"page"=>$page);
$string=array_merge($arrayimg,$dataval);
$data_string=json_encode($string);


//API URL
$url='https://localhost:5030/verifyPhoto';*/
//$data_string = file_get_contents("request.json");

$type = pathinfo($memPhoto, PATHINFO_EXTENSION);
$contents = file_get_contents($memPhoto);
$imgdata = 'data:image/' . $type . ';base64,'. base64_encode($contents);
//$regimage = $dataval['membership_no'].'_1_'.strtotime("now");
$source = $db;
$regimage = $photo_path;//'P'.$dataval['membership_no'];
$camimage = $dataval['membership_no'].'_2_'.strtotime("now");
$arrayimg = array("img1"=>$imgdata,"page"=>$page,"source"=>$source,"regimage"=>$regimage,"camimage"=>$camimage);
$string=array_merge($arrayimg,$dataval);
$data_string=json_encode($string);

if($page=='login'){
	$url='https://638b-202-144-77-234.ngrok-free.app/imgtest';
}else{
	$url='https://638b-202-144-77-234.ngrok-free.app/imgtest';
}

$ch = curl_init($url);
curl_setopt($ch, CURLOPT_POST, 1);
curl_setopt($ch, CURLOPT_POSTFIELDS, $data_string);
curl_setopt($ch, CURLOPT_SSL_VERIFYPEER, false); // To allow HTTPS connection
curl_setopt($ch, CURLOPT_SSL_VERIFYHOST, false);
curl_setopt($ch, CURLOPT_HTTPHEADER, array('Content-Type:application/json'));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, 1); 
curl_setopt($ch, CURLOPT_FAILONERROR, true);
$cur_res   = curl_exec($ch);
if( $cur_errno = curl_errno($ch) )	{
	$cur_errmsg= curl_strerror($cur_errno);
	header($_SERVER['SERVER_PROTOCOL'] . ' ' . $cur_errmsg, true, 500);
	echo $cur_errmsg;
	exit;
}
		
		

curl_close($ch);
 
//After getting response have to insert

if ($cur_res){
	$data = json_decode($data_string, true);
	//print_r($data);
	if(is_array($data)){  
		   $img1 = $data['img1'];
		   $img2 = $data['img2'];
		   $membership_no = $data['membership_no'];
		   $subject_code = $data['subject_code'];
		   if(isset($data['cam_id']['label'])) {
			$cam_id = $data['cam_id']['label'];
		   } else {
			$cam_id = 'No Data';
		   }
		   
		   $event_source_id = $data['event_source_id'];
			   
			}else {
				echo 'Can not able to read a JSON Format. Kindly check the format.';
			}
$curldata = json_decode($cur_res,true); 
$Accuracy = $curldata['Facial_distance'];
$no_of_faces = $curldata['no_of_faces'];
$detectionStatus = $curldata['error_code'];
$Accuracy = is_numeric($Accuracy)? $Accuracy : 0;
/* if($no_of_faces > 1) {
	$detectionStatus = 3;
} else if($no_of_faces < 1) {
	$detectionStatus = 4;
} else if($detectionStatus == 'Photo Matched') {
	$detectionStatus = 1;
} else if($detectionStatus == 'Not Matched') {
	$detectionStatus = 2;
} else {
	$detectionStatus = 2;
} */
//$detectionStatus = 1;
$candidatedata="select exam_time,exam_date,exam_code,subject_code,membership_no from iib_candidate_iway where  membership_no= '$membership_no' ";

$candidate_res = mysql_query($candidatedata, $SLAVECONN) or errorlog('err055', " QUERY: $candidatedata  " . mysql_error($SLAVECONN));
$row = mysql_fetch_assoc($candidate_res);
    $membership_no= $row['membership_no'];
    $exam_time= $row['exam_time'];
    $exam_date= $row['exam_date'];
    $exam_code= $row['exam_code'];
    $subject_code= $row['subject_code'];
	live_tracking($membership_no,$exam_time,$exam_date,$exam_code,$subject_code,$Accuracy,$no_of_faces,$detectionStatus,$img1,$img2,$cam_id,$event_source_id);
}

function live_tracking($membership_no,$exam_time,$exam_date,$exam_code,$subject_code,$Accuracy,$no_of_faces,$detectionStatus,$img1,$img2,$cam_id,$event_source_id){
	global $MASTERCONN;	
	global $SLAVECONN;
	$ipAddress = $_SERVER['REMOTE_ADDR'];
	$mac_id = getMacAddress();
	if($detectionStatus !== 1) {
		$livetracking = "live_tracking";
		$imagerepository = "image_repository";
		$imagerepouri = "image_repository_uri";
	} else {
		$livetracking = "live_tracking_success";
		$imagerepository = "image_repository_success";
		$imagerepouri = "image_repository_uri_success";
	}
	$countdata="select count(1) as count,id from $livetracking where  membership_no= '$membership_no' and subject_code='$subject_code' and exam_time='$exam_time' and event_source_id ='$event_source_id';";
	$get_select_count = mysql_query($countdata, $SLAVECONN) or errorlog('err054', " QUERY: $countdata  " . mysql_error($SLAVECONN));
	$rescount=mysql_fetch_assoc($get_select_count);
	$livetrack_last_id =$rescount['id'];
	
		if($rescount['count'] == 0){
			$livetrackInsert = "INSERT INTO $livetracking (exam_time, exam_date, exam_code, subject_code, membership_no, mac_id, event_source_id) VALUES ('$exam_time', '$exam_date', '$exam_code', '$subject_code','$membership_no', '$mac_id','$event_source_id'); ";
			$liveinsert = mysql_query($livetrackInsert, $SLAVECONN) or errorlog('err052', " QUERY: $livetrackInsert  " . mysql_error($SLAVECONN));
			$livetrack_last_id = mysql_insert_id($SLAVECONN);
		}
			$image_repository = "INSERT INTO $imagerepository (live_tracking_id, cam_id, ip, status, integrity_score, no_of_faces, timestamp) VALUES ('$livetrack_last_id', '$cam_id', '$ipAddress', '$detectionStatus','$Accuracy', $no_of_faces, now()); ";
			$imagerepinsert = mysql_query($image_repository, $SLAVECONN) or errorlog('err051', " QUERY: $image_repository  " . mysql_error($SLAVECONN));
			$image_repo_lastid = mysql_insert_id($SLAVECONN);
			$image_repository_uri_img2 = "INSERT INTO $imagerepouri (image_repository_id, proctor_image_uri, image_source) VALUES ('$image_repo_lastid', '$img2','3'); ";
	       $image_repository_uri_insert_img2 = mysql_query($image_repository_uri_img2, $SLAVECONN) or errorlog('err053', " QUERY: $image_repository_uri_img2  " . mysql_error($SLAVECONN));
		 
}
echo $cur_res;
?>
