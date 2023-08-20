import 'antd/dist/antd.css';
import React from 'react';
import { BrowserRouter, Routes, Route, Link, Navigate } from 'react-router-dom';
// import DiagnosisPage from './pages/DiagnosisPage';
// import WhiteLightImagePage from './pages/WhiteLightImagePage';
import EndoscopicUltrasoundImagePage from './pages/EndoscopicUltrasoundImagePage';
// import HistoricalModelPage from './pages/HistoricalModelPage';
// import LatestModelNotificationPage from './pages/LatestModelNotificationPage';
// import Neo4jVis from './pages/Neo4jVis';
import { Login } from './login/Login';

import { Layout, Menu } from 'antd';
const { SubMenu } = Menu;
const { Header, Content, Sider } = Layout;

function PageLayout({ children }) {
    return (
        <Layout style={{ height: '100vh', display: 'flex' }}>
            <Header className="site-layout-background" style={{ padding: 0 }}>
                <Menu mode="horizontal" defaultSelectedKeys={['/diagnosis']}>
                    {/* <Menu.Item key="/white-light-image">
                        <Link to="/white-light-image">白光图像</Link>
                    </Menu.Item> */}
                    {/* <Menu.Item key="/endoscopic-ultrasound-image"> */}
                    {/* <Link to="/endoscopic-ultrasound-image">超声图像</Link> */}
                    {/* </Menu.Item> */}
                    {/* <Menu.Item key="/medical-graph"> */}
                    {/* <Link to="/medical-graph">内镜超声知识图</Link> */}
                    {/* </Menu.Item> */}
                    <Menu.Item key="/diagnosis">
                        <Link to="/diagnosis">图像诊断</Link>
                    </Menu.Item>
                    {/* <Menu.Item key="/historical-model"> */}
                    {/* <Link to="/historical-model">模型优化</Link> */}
                    {/* </Menu.Item> */}
                    {/* <Menu.Item key="/latest-model-notification"> */}
                    {/* <Link to="/latest-model-notification">模型通知</Link> */}
                    {/* </Menu.Item> */}
                </Menu>
            </Header>

            <Content
                className="site-layout-background"
                style={{
                    padding: 24,
                    margin: 0,
                    height: '100%',
                }}
            >
                {children}
            </Content>
        </Layout>
    );
}

function App() {
    return (
        <BrowserRouter>
            <Routes>
                <Route path="/" element={<Login />} />
                <Route
                    path="/diagnosis"
                    element={
                        <PageLayout>
                            <DiagnosisPage />
                        </PageLayout>
                    }
                />
                {/* <Route
                    path="/white-light-image"
                    element={
                        <PageLayout>
                            <WhiteLightImagePage />
                        </PageLayout>
                    }
                />
                <Route
                    path="/endoscopic-ultrasound-image"
                    element={
                        <PageLayout>
                            <EndoscopicUltrasoundImagePage />
                        </PageLayout>
                    }
                />
                <Route
                    path="/historical-model"
                    element={
                        <PageLayout>
                            <HistoricalModelPage />
                        </PageLayout>
                    }
                />
                <Route
                    path="/latest-model-notification"
                    element={
                        <PageLayout>
                            <LatestModelNotificationPage />
                        </PageLayout>
                    }
                />
                <Route
                    path="/medical-graph"
                    element={
                        <PageLayout>
                            <Neo4jVis />
                        </PageLayout>
                    }
                /> */}
                <Route path="/" element={<Navigate to="/diagnosis" />} />
            </Routes>
        </BrowserRouter>
    );
}

export default App;
