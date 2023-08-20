import React, { useState } from 'react';
import { Row, Col, Layout, Typography, Spin, Card, Button } from 'antd';
import { UploadOutlined } from '@ant-design/icons';

function ImageUpload({ onUpload }) {
    const [file, setFile] = useState(null);
    const [imageUrl, setImageUrl] = useState(null);
    const [loading, setLoading] = useState(false); // 定义loading状态

    const handleUpload = () => {
        const formData = new FormData();
        formData.append('image', file);

        setLoading(true); // 设置loading状态为true

        return fetch('http://1.15.55.32:8002/upload/', {
            method: 'POST',
            body: formData,
            mode: 'cors',
        })
            .then(response => response.json())
            .then(data => {
                setLoading(false); // 请求完成后设置loading状态为false
                setImageUrl(data.url);
                onUpload(data.url);
            })
            .catch(error => {
                setLoading(false); // 请求完成后设置loading状态为false
                console.error(error);
            });
    };

    return (
        <div style={{ display: 'flex', justifyContent: 'flex-start', flexDirection: 'row' }}>
            <input type="file" onChange={e => setFile(e.target.files[0])} />
            <Button
                style={{ width: 110, height: 35, marginLeft: 10 }}
                onClick={handleUpload}
                type="default"
                icon={<UploadOutlined />}
            >
                图像上传
            </Button>
            {loading && (
                <Spin>
                    <span>上传中...</span>
                </Spin>
            )}
        </div>
    );
}

export default ImageUpload;
