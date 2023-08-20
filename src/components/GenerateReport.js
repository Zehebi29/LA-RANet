import React, { useState, useEffect } from 'react';
import { message, Card, Button } from 'antd';
import { PlusSquareOutlined } from '@ant-design/icons';
import MySelect from './MySelect';

function GenerateReport({ image, prediction, predictionProbs, isFileSelected, clearTextArea }) {
    const options1 = [
        { value: '1', label: '食管' },
        { value: '2', label: '胃窦' },
        { value: '3', label: '胃体' },
        { value: '4', label: '胃底' },
        { value: '5', label: '贲门' },
        { value: '6', label: '十二指肠' },
        { value: '7', label: '升结肠' },
    ];
    const options2 = [
        { value: '1', label: '高' },
        { value: '2', label: '中' },
        { value: '3', label: '低' },
        { value: '4', label: '混合' },
    ];
    const options3 = [
        { value: '1', label: '粘膜层' },
        { value: '2', label: '粘膜肌层' },
        { value: '3', label: '粘膜下层' },
        { value: '4', label: '固有肌层' },
    ];
    const [textAreaValue, setTextAreaValue] = useState('');
    const [selectedOption1, setSelectedOption1] = useState('');
    const [selectedOption2, setSelectedOption2] = useState('');
    const [selectedOption3, setSelectedOption3] = useState('');

    const handleTextAreaChange = event => {
        setTextAreaValue(event.target.value);
    };

    const handleOptionChange1 = selectedOption => {
        setSelectedOption1(selectedOption);
        console.log('选项：', selectedOption1);
    };

    const handleOptionChange2 = options2 => {
        setSelectedOption2(options2);
    };

    const handleOptionChange3 = options3 => {
        setSelectedOption3(options3);
    };

    const handleSubmit = async event => {
        event.preventDefault();

        try {
            // 将请求参数改为符合后端结构体的形式
            const label1 = options1.find(option => option.value === selectedOption1)?.label;
            const label2 = options2.find(option => option.value === selectedOption2)?.label;
            const label3 = options3.find(option => option.value === selectedOption3)?.label;
            const imageName = image.split('/').pop();
            console.log(imageName);

            const reportRequest = {
                img_name: imageName,
                textAreaValue: textAreaValue,
                selectedOption1: label1,
                selectedOption2: label2,
                selectedOption3: label3,
            };
            console.log(reportRequest);

            // 使用 fetch API 发送表单数据到 FastAPI 后端
            const response = await fetch('http://1.15.55.32:8003/report', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(reportRequest),
            });

            // 解析响应并检查是否成功提交表单
            const data = await response.json();
            if (data.success) {
                // 执行提交成功动画
                message.success('提交成功');
            } else {
                // 执行提交失败动画
                message.error('提交失败');
            }
        } catch (error) {
            // 执行提交失败动画
            message.error('提交失败');
        }
    };

    useEffect(() => {
        if (!isFileSelected) {
            setTextAreaValue('');
            clearTextArea();
        }
    }, [isFileSelected, clearTextArea]);

    return (
        <Card title="EUS 图像诊断报告" style={{ marginTop: 30, height: 300 }}>
            <div style={{ display: 'flex', flexDirection: 'row' }}>
                <div style={{ flex: 1 }}>
                    {/* 左边区域 */}

                    <p style={{ fontSize: 14 }}>预测类别：{prediction}</p>
                    <p style={{ fontSize: 14 }}>置信度为：{predictionProbs}</p>
                    <div
                        style={{
                            border: 20,
                            borderColor: 'yellow',
                        }}
                    >
                        <p>
                            病变部位：{' '}
                            <MySelect
                                options={options1}
                                value={selectedOption1}
                                onChange={handleOptionChange1}
                            />
                        </p>
                        <p>
                            回声高低：{' '}
                            <MySelect
                                options={options2}
                                value={selectedOption2}
                                onChange={handleOptionChange2}
                            />
                        </p>
                        <p>
                            起源层次：{' '}
                            <MySelect
                                options={options3}
                                value={selectedOption3}
                                onChange={handleOptionChange3}
                            />
                        </p>
                    </div>
                </div>
                <div style={{ flex: 1, overflow: 'hidden' }}>
                    {/* 右边区域 */}
                    <p>
                        其他补充信息：
                        <textarea
                            value={textAreaValue}
                            onChange={handleTextAreaChange}
                            style={{
                                // width: '100%',
                                height: 115,
                                width: '100%',
                                overflow: 'hidden',
                            }}
                            placeholder="请输入其他补充信息，如正确的疾病类别"
                        />
                    </p>
                    <div
                        style={{
                            display: 'flex',
                            flexDirection: 'column',
                        }}
                    >
                        <Button
                            type="ghost"
                            onClick={handleSubmit}
                            style={{
                                width: '100%',
                            }}
                            icon={<PlusSquareOutlined />}
                        >
                            提交报告
                        </Button>
                    </div>
                </div>
            </div>
        </Card>
    );
}

export default GenerateReport;
