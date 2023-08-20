import React, { useState } from 'react';
import {
    Image,
    Select,
    Input,
    Layout,
    Typography,
    message,
    Row,
    Col,
    Card,
    Skeleton,
    Button,
    notification,
} from 'antd';
import ImageUpload from '../components/ImageUpload';
import ModelPredict from '../components/ModelPredict';

import GenerateReport from '../components/GenerateReport';
import { PlusSquareOutlined } from '@ant-design/icons';
import { LeftOutlined, RightOutlined } from '@ant-design/icons';

const { Option } = Select;
const { Content } = Layout;
const { Title, Paragraph } = Typography;
let id = 0;
const styles = {
    imageContainer: {
        borderWidth: '8px',
        borderStyle: 'solid',
        borderColor: 'black',
    },
    category1: {
        borderColor: 'red',
    },
    category2: {
        borderColor: 'green',
    },
    category3: {
        borderColor: 'blue',
    },
    category4: {
        borderColor: 'yellow',
    },
    category5: {
        borderColor: 'purple',
    },
};

function DiagnosisPage() {
    const [imageUrl, setImageUrl] = useState(null);
    const [uploadedImages, setUploadedImages] = useState([]);
    const [uploadedLocImages, setUploadedLocImages] = useState([]);
    const [prediction, setPrediction] = useState(null);
    const [predictions, setPredictions] = useState(new Array(uploadedImages.length).fill(null));
    const [locPredictions, setLocPredictions] = useState(
        new Array(uploadedLocImages.length).fill(null)
    );
    const [predictionProbs, setPredicitonProbs] = useState(null);
    const [predictionImageUrl, setPredictionImageUrl] = useState(null);
    const [showResult, setShowResult] = useState(false);
    const [images, setImages] = useState([]);
    const [isFileSelected, setIsFileSelected] = useState(false);

    const categoryMappings = {
        平滑肌瘤: '1',
        神经内分泌肿瘤: '2',
        胃肠道间质瘤: '3',
        异位胰腺: '4',
        脂肪瘤: '5',
    };

    // const [selectedImage, setSelectedImage] = useState('');

    // const onImageSelect = value => {
    // console.log(value/);
    // };

    const handleClearTextArea = () => {
        setIsFileSelected(false);
    };

    const handleImageSelect = value => {
        // setSelectedImage(value);
        // onImageSelect(value);
    };

    const handleUpload = url => {
        id += 1;
        setImageUrl(url);
        setPrediction(null); // 图像更新后重置预测结果
        setPredicitonProbs(null);
        setPredictionImageUrl(null);
        setShowResult(false);
        setIsFileSelected(true);
        setImages([...images, { id, url }]);
        // setImages(pastImages => [...pastImages, url]);
    };

    const handlePredict = data => {
        setShowResult(true);
        if (data.cls && data.probs && data.url) {
            setPrediction(data.cls);
            setPredicitonProbs(data.probs);
            setPredictionImageUrl(data.url);
            console.log(predictionImageUrl);
            setShowResult(true);
            console.log('in handlePredict:', predictions);
        } else {
            message.error('预测失败，请重试！');
            setShowResult(false);
        }
    };

    const handleAddImage = url => {
        if (url) {
            setUploadedImages(prevImages => [...prevImages, url]);
        }
        setPredictions({ ...predictions, [uploadedImages.length]: prediction });
    };

    const handleAddLocImage = url => {
        if (url) {
            setUploadedLocImages(prevLocImages => [...prevLocImages, url]);
        }
        setLocPredictions({ ...locPredictions, [uploadedLocImages.length]: prediction });
    };

    const handleSubmit = event => {
        event.preventDefault();
        // 处理表单数据逻辑
        // ...

        // 执行提交成功动画
        message.success('提交成功');
    };

    const handleLeftClick = () => {
        message.success('已加入自监督学习库');
    };

    const handleRightClick = () => {
        message.success('已加入分类学习库');
    };

    const handleRemove = index => {
        setUploadedImages(prevLocImages => {
            const images = [...prevLocImages];
            images.splice(index, 1);
            return images;
        });
        if (Array.isArray(predictions)) {
            // 对数组进行迭代操作
            setPredictions(predictions => {
                const new_predictions = [...predictions];
                new_predictions.splice(index, 1);
                return new_predictions;
            });
            console.log('在handleRemove中：', index, predictions, 'handleRemove结束');
        } else {
            setPredictions(predictions => {
                const newPredictions = {};
                for (const key in predictions) {
                    if (key !== String(index)) {
                        newPredictions[key] = predictions[key];
                    }
                }
                return newPredictions;
            });
        }
    };

    const handleLocRemove = index => {
        setUploadedLocImages(prevLocImages => {
            const images = [...prevLocImages];
            images.splice(index, 1);
            return images;
        });
        if (Array.isArray(locPredictions)) {
            // 对数组进行迭代操作
            setLocPredictions(locPredictions => {
                const newLocPredictions = [...locPredictions];
                newLocPredictions.splice(index, 1);
                return newLocPredictions;
            });
            console.log('在handleRemove中：', index, locPredictions, 'handleRemove结束');
        } else {
            setLocPredictions(locPredictions => {
                const newLocPredictions = {};
                for (const key in locPredictions) {
                    if (key !== String(index)) {
                        newLocPredictions[key] = locPredictions[key];
                    }
                }
                return newLocPredictions;
            });
        }
    };

    return (
        <Layout>
            {/* 头部 */}
            <Content style={{ backgroundColor: '#008080' }}>
                <div className="header-banner">
                    <div style={{ margin: '10px 20px' }} className="header-banner-title">
                        <div style={{ display: 'flex', alignItems: 'center' }}>
                            {/* 放图标 */}
                            {/* <Image
                                preview={false}
                                src="dhu_icon.png"
                                width={50}
                                height={50}
                                alt="My Icon"
                            /> */}
                            <Title style={{ color: '#FFFFFF', fontSize: 18, marginLeft: 10 }}>
                                算法库：自监督预训练分类模型
                            </Title>
                        </div>
                    </div>
                </div>
            </Content>

            {/* 内容 */}
            <Content>
                <Row gutter={[32, 32]} style={{ marginBlockStart: 32, marginBottom: 32 }}>
                    {/* 第一个区域 */}
                    <Col xs={24} sm={24} md={12} lg={12} xl={12}>
                        <Card
                            bordered={false}
                            style={{ backgroundColor: '#FFFFFF', height: '100%' }}
                        >
                            <div
                                style={{ display: 'flex', flexDirection: 'column', marginTop: 16 }}
                            >
                                {/* 区域一：标题栏 */}
                                <div style={{ marginBottom: 5 }}>
                                    <Title level={3} style={{ fontSize: 18, color: 'teal' }}>
                                        上传图像
                                    </Title>
                                </div>
                                {/* 区域一：功能按键 */}
                                <div
                                    style={{
                                        display: 'flex',
                                        flexDirection: 'row',
                                    }}
                                >
                                    <div style={{ flex: 1 }}>
                                        <ImageUpload onUpload={handleUpload} />
                                    </div>
                                    <div
                                        style={{
                                            flex: 1,
                                            display: 'flex',
                                            flexDirection: 'row',
                                            overflow: 'hidden',
                                        }}
                                    >
                                        <Button
                                            type="primary"
                                            shape="circle"
                                            style={{
                                                marginLeft: 160,
                                            }}
                                            icon={<LeftOutlined />}
                                            onClick={handleLeftClick}
                                        />
                                        <Button
                                            type="primary"
                                            shape="circle"
                                            style={{
                                                marginLeft: 60,
                                            }}
                                            icon={<RightOutlined />}
                                            onClick={handleRightClick}
                                        />
                                    </div>
                                </div>
                                {/* 区域一：结果展示 */}
                                <Row gutter={[16, 16]}>
                                    {/* 区域一：展示区一：上传图像，包含历史 */}
                                    <Col span={12}>
                                        {imageUrl ? (
                                            <Row
                                                gutter={[16, 16]}
                                                style={{
                                                    height: 300,
                                                    overflowY: 'scroll',
                                                    marginTop: 20,
                                                }}
                                            >
                                                {images.map(image => (
                                                    <Col key={image.id} span={8}>
                                                        <Image
                                                            preview={false}
                                                            src={image.url}
                                                            onClick={() => handleImageSelect(image)}
                                                            style={{
                                                                cursor: 'pointer',
                                                                marginTop: 10,
                                                            }}
                                                        />
                                                    </Col>
                                                ))}
                                            </Row>
                                        ) : (
                                            <div style={{ textAlign: 'left', marginTop: 50 }}>
                                                <Title style={{ fontSize: 18, color: '#666' }}>
                                                    请上传EUS图像
                                                </Title>
                                                <Paragraph style={{ color: '#666' }}>
                                                    支持格式：JPG, JPEG, PNG, BMP
                                                </Paragraph>
                                            </div>
                                        )}
                                    </Col>
                                    <Col span={12}>
                                        {imageUrl ? (
                                            <div style={{ marginTop: 30, textAlign: 'center' }}>
                                                <img
                                                    src={imageUrl}
                                                    alt="上传的图像"
                                                    style={{
                                                        height: 300,
                                                        maxWidth: '100%',
                                                    }}
                                                />
                                            </div>
                                        ) : (
                                            <div style={{ textAlign: 'left', marginTop: 50 }}>
                                                <Title style={{ fontSize: 18, color: '#666' }}>
                                                    图像展示区域
                                                </Title>
                                                <Paragraph style={{ color: '#666' }}>
                                                    左侧会显示历史上传图像
                                                </Paragraph>
                                            </div>
                                        )}
                                    </Col>
                                    {/* 区域一：展示区二：图像显示 */}
                                </Row>
                            </div>
                        </Card>
                    </Col>

                    {/* 第二个区域 */}
                    <Col xs={24} sm={24} md={12} lg={12} xl={12}>
                        <Card
                            bordered={false}
                            style={{ backgroundColor: '#FFFFFF', height: '100%' }}
                        >
                            <div
                                style={{ display: 'flex', flexDirection: 'column', marginTop: 16 }}
                            >
                                {/* 区域二：标题栏 */}
                                <div>
                                    <Title level={3} style={{ fontSize: 18, color: 'teal' }}>
                                        诊断结果
                                    </Title>
                                </div>
                                {/* 区域二：功能按键 */}
                                <div
                                    style={{
                                        display: 'flex',
                                        flexDirection: 'row',
                                    }}
                                >
                                    <div
                                        style={{
                                            flex: 1,
                                            display: 'flex',
                                            flexDirection: 'row',
                                            // width: '100%',
                                        }}
                                    >
                                        <div style={{ flex: 1 }}>
                                            <ModelPredict
                                                image={imageUrl}
                                                onPredict={handlePredict}
                                            />
                                        </div>
                                        <div style={{ flex: 1 }}>
                                            <Button
                                                type="dashed"
                                                danger
                                                onClick={() => handleAddImage(imageUrl)}
                                                style={{
                                                    width: 110,
                                                    height: 35,
                                                }}
                                                icon={<PlusSquareOutlined />}
                                            >
                                                预测错误
                                            </Button>
                                        </div>
                                        <div style={{ flex: 1 }}>
                                            <Button
                                                type="dashed"
                                                danger
                                                onClick={() => handleAddLocImage(imageUrl)}
                                                style={{
                                                    width: 110,
                                                    height: 35,
                                                }}
                                                icon={<PlusSquareOutlined />}
                                            >
                                                定位错误
                                            </Button>
                                        </div>
                                    </div>
                                    <div style={{ flex: 1 }}></div>
                                </div>
                                {/* 区域二：结果展示 */}
                                <Row gutter={[16, 16]}>
                                    {/* 区域二：展示区一：上传图像 */}
                                    <Col span={12}>
                                        {prediction ? (
                                            <div></div>
                                        ) : (
                                            showResult && (
                                                <div style={{ textAlign: 'left' }}>
                                                    <Title level={3} style={{ fontSize: 18 }}>
                                                        正在处理图像，请稍等...
                                                    </Title>
                                                    <Skeleton active />
                                                </div>
                                            )
                                        )}
                                        <div
                                            style={{
                                                display: 'flex',
                                                flexDirection: 'column',
                                            }}
                                        >
                                            {predictionImageUrl ? (
                                                // <div>
                                                //     <img
                                                //         src={predictionImageUrl}
                                                //         alt="预测的图像"
                                                //         style={{ maxWidth: '100%' }}
                                                //     />
                                                // </div>
                                                <div style={{ marginTop: 30, textAlign: 'center' }}>
                                                    <img
                                                        src={predictionImageUrl}
                                                        alt="预测的图像"
                                                        style={{
                                                            height: 300,
                                                            maxWidth: '100%',
                                                        }}
                                                    />
                                                </div>
                                            ) : (
                                                <div style={{ textAlign: 'left', marginTop: 50 }}>
                                                    <Title style={{ fontSize: 18, color: '#666' }}>
                                                        预测结果显示
                                                    </Title>
                                                    <Paragraph style={{ color: '#666' }}>
                                                        高亮部分为模型决策依据
                                                    </Paragraph>
                                                </div>
                                            )}
                                        </div>
                                    </Col>
                                    {/* 区域二：展示区二：待定 */}
                                    <Col span={12}>
                                        <GenerateReport
                                            image={imageUrl}
                                            prediction={prediction}
                                            predictionProbs={predictionProbs}
                                            isFileSelected={isFileSelected}
                                            clearTextArea={handleClearTextArea}
                                        />
                                    </Col>
                                </Row>
                            </div>
                        </Card>
                    </Col>
                </Row>

                <Row gutter={[32, 32]} style={{ marginBottom: 32 }}>
                    {/* 第三个区域 */}
                    <Col xs={24} sm={24} md={12} lg={12} xl={12}>
                        <Card
                            bordered={false}
                            style={{ backgroundColor: '#FFFFFF', height: '100%' }}
                        >
                            <Title level={3} style={{ fontSize: 18, color: 'teal' }}>
                                提示
                            </Title>
                            <div>
                                <div style={{ textAlign: 'justify' }}>
                                    <img src="note1.jpg" alt="My Image" style={{ width: '100%' }} />
                                </div>
                            </div>
                        </Card>
                    </Col>
                    {/* 第四个区域 */}
                    <Col xs={24} sm={24} md={12} lg={12} xl={12}>
                        <Card
                            bordered={false}
                            style={{ backgroundColor: '#FFFFFF', height: '100%' }}
                        >
                            <Title level={3} style={{ fontSize: 18, color: 'teal' }}>
                                提示
                            </Title>
                            <div style={{ textAlign: 'justify' }}>
                                <img src="note2.jpg" alt="My Image" style={{ width: '100%' }} />
                                <div className="radar-chart"></div>
                            </div>
                        </Card>
                    </Col>
                </Row>
            </Content>

            {/* 底部1 */}
            <Content style={{ backgroundColor: '#008080' }}>
                <div className="footer-banner">
                    <div className="footer-banner-title" style={{ margin: '10px 20px' }}>
                        <Title style={{ color: '#FFFFFF', fontSize: 18, marginLeft: 10 }}>
                            病例库：类别预测错误
                        </Title>
                    </div>
                    <div className="footer-banner-predict"></div>
                </div>
            </Content>

            <Content style={{ height: 400 }}>
                <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center' }}>
                    {uploadedImages.map((url, index) => {
                        let buttonRef = null;
                        const cls = predictions[index];
                        return (
                            <div
                                key={index}
                                style={{
                                    margin: 8,
                                    position: 'relative',
                                    ...styles.imageContainer,
                                    ...(cls && styles[`category${categoryMappings[cls]}`]),
                                }}
                            >
                                <img
                                    src={url}
                                    // onLoad={() => console.log(index, cls, categoryMappings[cls])}
                                    alt={`已上传的图像${index + 1}`}
                                    height={150}
                                />
                                <Button
                                    type="primary"
                                    size="small"
                                    style={{
                                        position: 'absolute',
                                        top: 8,
                                        right: 8,
                                        display: 'block',
                                    }}
                                    onMouseEnter={() => (buttonRef.style.display = 'block')}
                                    onMouseLeave={() => (buttonRef.style.display = 'block')}
                                    onClick={() => handleRemove(index)}
                                    ref={ref => (buttonRef = ref)}
                                >
                                    删除
                                </Button>
                            </div>
                        );
                    })}
                </div>
            </Content>
            {/* 底部2 */}
            <Content style={{ backgroundColor: '#008080' }}>
                <div className="footer-banner">
                    <div className="footer-banner-title" style={{ margin: '10px 20px' }}>
                        <Title style={{ color: '#FFFFFF', fontSize: 18, marginLeft: 10 }}>
                            病例库：CAM定位错误
                        </Title>
                    </div>
                    <div className="footer-banner-predict"></div>
                </div>
            </Content>

            <Content style={{ height: 400 }}>
                <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center' }}>
                    {uploadedLocImages.map((url, index) => {
                        let buttonRef = null;
                        const cls = locPredictions[index];
                        return (
                            <div
                                key={index}
                                style={{
                                    margin: 8,
                                    position: 'relative',
                                    ...styles.imageContainer,
                                    ...(cls && styles[`category${categoryMappings[cls]}`]),
                                }}
                            >
                                <img
                                    src={url}
                                    // onLoad={() => console.log(index, cls, categoryMappings[cls])}
                                    alt={`已上传的图像${index + 1}`}
                                    height={150}
                                />
                                <Button
                                    type="primary"
                                    size="small"
                                    style={{
                                        position: 'absolute',
                                        top: 8,
                                        right: 8,
                                        display: 'block',
                                    }}
                                    onMouseEnter={() => (buttonRef.style.display = 'block')}
                                    onMouseLeave={() => (buttonRef.style.display = 'block')}
                                    onClick={() => handleLocRemove(index)}
                                    ref={ref => (buttonRef = ref)}
                                >
                                    删除
                                </Button>
                            </div>
                        );
                    })}
                </div>
            </Content>
        </Layout>
    );
}
export default DiagnosisPage;
