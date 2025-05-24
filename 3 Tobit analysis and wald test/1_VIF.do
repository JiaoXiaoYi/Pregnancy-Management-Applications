import delimited "/Users/heyunfan/Desktop/女性健康app/2数据收集/2第二版本-删掉平台评分为0的app/第二版本-6-stata-输入-data_result7.csv" , varnames(1) clear   //录入数据
reg nd x1-x12
vif
