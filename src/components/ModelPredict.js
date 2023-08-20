import React, { useState } from 'react';
import { Button } from 'antd';
import axios from 'axios';
import { CheckCircleOutlined } from '@ant-design/icons';

function ModelPredict({ image, onPredict }) {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const predict = () => {
        setLoading(true);
        const img_name = image.split('/').pop(); // 解析出图片文件名
        const imagePath = `upload/${img_name}`; // 新路径参数
        const formData = new FormData();
        formData.append('img_path', imagePath);

        // const imagePath = `predict/test.jpg`; // 新路径参数
        console.log(imagePath);
        return fetch('http://1.15.55.32:8002/predict/', {
            method: 'POST',
            body: formData,
        })
            .then(response => {
                console.log('Response:', response);
                return response.json();
            })
            .then(data => {
                console.log('prediction result:', data);
                onPredict(data);
            })
            .catch(error => console.error(error))
            .finally(() => setLoading(false));
    };

    return (
        <div>
            <Button
                type="dashed"
                onClick={predict}
                icon={<CheckCircleOutlined />}
                style={{ width: 110, height: 35 }}
            >
                模型预测
            </Button>
            {loading && (
                <div style={{ marginLeft: 10 }}>
                    <span style={{ color: 'blue' }}>加载中...</span>
                </div>
            )}

            {error && <div style={{ marginLeft: 10, color: 'red' }}>请求出错，请稍后再试。</div>}
        </div>
    );
}

export default ModelPredict;
