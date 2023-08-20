// 登录界面

import React, { useState, useEffect, useRef } from 'react';
import './Login.css';
import { message } from 'antd';
import { useNavigate } from 'react-router-dom';
import { server_url } from './UrlManagement';

export function Login() {
    const [flag, setFlag] = useState(true);
    const [username, setUsername] = useState(undefined);
    const [password, setPassword] = useState(undefined);
    const [passwordAgain, setPasswordAgain] = useState(undefined);
    const canvasRef = useRef();
    const navigate = useNavigate();

    const changeClassName = () => {
        setFlag(!flag);
        // container.className.add("right-panel-active")
    };
    const changeUsername = e => {
        setUsername(e.target.value);
    };
    const changePassword = e => {
        setPassword(e.target.value);
    };
    const changePasswordAgain = e => {
        setPasswordAgain(e.target.value);
    };
    const signUp = () => {
        if (!username) {
            message.warning('请输入用户名');
            return;
        }
        if (!password) {
            message.warning('请输入密码');
            return;
        }
        if (!passwordAgain) {
            message.warning('请再次输入密码');
            return;
        }
        if (passwordAgain !== password) {
            message.warning('两次密码输入不一致');
            return;
        }
        let user = {
            userID: username,
            password: password,
        };
        // const localurl = "http://localhost:8001/register/";
        const serverurl = server_url;
        fetch(serverurl, {
            method: 'post',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(user),
        })
            .then(data => {
                return data.json();
            })
            .then(data => {
                if (data.state) {
                    message.success(data.message);
                } else {
                    message.warning(data.message);
                }
            });
    };
    const signIn = () => {
        if (!username) {
            message.warning('请输入用户名');
            return;
        }
        if (!password) {
            message.warning('请输入密码');
            return;
        }
        const user = {
            userID: username,
            password: password,
        };
        // if (user.userID === "gaojun" && user.password === "gaojun") {
        //   message.success("登录成功");
        //   sessionStorage.setItem("username", user.userID);
        //   props.history.push("/"); //路由跳转
        // } else {
        //     message.warning("请检查用户名或密码是否正确");
        // }
        fetch('http://81.68.124.174:30545/DTE_login/user', {
            method: 'post',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(user),
        })
            .then(data => {
                return data.json();
            })
            .then(data => {
                if (data.state) {
                    // 清除canvas元素
                    canvasRef.current.parentNode.removeChild(canvasRef.current);
                    message.success(data.message);
                    sessionStorage.setItem('username', user.userID);
                    //   eventService.nameStorageEmit(user.userID);
                    navigate('/diagnosis'); //路由跳转
                } else {
                    message.warning('请检查用户名或密码是否正确');
                }
            });
    };
    useEffect(() => {
        if (sessionStorage.getItem('username2')) {
            sessionStorage.removeItem('username2');
        }
    }, []);

    useEffect(() => {
        //canvas元素相关
        //创建canvas元素，并设置canvas元素的id
        var canvas = canvasRef.current,
            context = canvas.getContext('2d'),
            attr = getAttr();
        let W = null;
        let H = null;

        //设置创建的canvas的相关属性
        canvas.id = 'c_n' + attr.length;
        canvas.style.cssText =
            'position:fixed;top:0;left:0;z-index:' + attr.z + ';opacity:' + attr.opacity;
        //将canvas元素添加到body元素中
        document.getElementsByTagName('body')[0].appendChild(canvas);
        //该函数设置了canvas元素的width属性和height属性
        getWindowWH();
        //onresize 事件会在窗口或框架被调整大小时发生
        //此处即为当窗口大小改变时，重新获取窗口的宽高和设置canvas元素的宽高
        window.onresize = getWindowWH;

        //该函数会得到引用了本文件的script元素，
        //因为本文件中在赋值时执行了一次getScript函数，html文件引用本文件时，本文件之后的script标签还没有被浏览器解释，
        //所以得到的script数组中，引用了本文的script元素在该数组的末尾
        //该函数的用意为使开发者能直接修改在html中引入该文件的script元素的属性来修改画布的一些属性，画布的z-index，透明度和小方块数量，颜色
        //与前面往body元素添加canvas元素的代码配合，当开发者想要使用该特效作为背景时，只需在html文件中添加script元素并引用本文件即可
        function getAttr() {
            let scripts = document.getElementsByTagName('script'),
                len = scripts.length,
                script = scripts[len - 1]; //v为最后一个script元素，即引用了本文件的script元素
            return {
                length: len,
                z: script.getAttribute('zIndex') || 0,
                opacity: script.getAttribute('opacity') || 0.8,
                color: script.getAttribute('color') || '255, 255, 255',
                count: script.getAttribute('count') || 99,
            };
        }

        //获得窗口宽高，并设置canvas元素宽高
        function getWindowWH() {
            // eslint-disable-next-line no-unused-expressions
            (W = canvas.width =
                window.innerWidth ||
                document.documentElement.clientWidth ||
                document.body.clientWidth),
                (H = canvas.height =
                    window.innerHeight ||
                    document.documentElement.clientHeight ||
                    document.body.clientHeight);
        }

        //生成随机位置的小方块
        var random = Math.random,
            squares = []; //存放小方块

        //往squares[]数组放小方块
        for (let p = 0; p < attr.count; p++) {
            var square_x = random() * W, //横坐标
                square_y = random() * H, //纵坐标
                square_xa = 2 * random() - 1, //x轴位移 -1,1
                square_ya = 2 * random() - 1; //y轴位移

            squares.push({
                x: square_x,
                y: square_y,
                xa: square_xa,
                ya: square_ya,
                max: 6000,
            });
        }
        //生成鼠标小方块

        var mouse = {
            x: null,
            y: null,
            max: 20000,
        };

        //获取鼠标所在坐标
        window.onmousemove = function (i) {
            //i为W3C DOM，window.event 为 IE DOM，以实现兼容IE
            //不过目前似乎IE已经支持W3C DOM，我用的是IE11，我注释掉下一句代码也能实现鼠标交互效果，
            //网上说7/8/9是不支持的，本人没有试验，
            //当然加上是没有错的
            i = i || window.event;
            mouse.x = i.clientX;
            mouse.y = i.clientY;
        };

        //鼠标移出窗口后，消除鼠标小方块

        window.onmouseout = function () {
            mouse.x = null;
            mouse.y = null;
        };

        //绘制小方块，小方块移动(碰到边界反向移动)，小方块受鼠标束缚
        var animation =
            window.requestAnimationFrame ||
            window.webkitRequestAnimationFrame ||
            window.mozRequestAnimationFrame ||
            window.oRequestAnimationFrame ||
            window.msRequestAnimationFrame ||
            function (i) {
                window.setTimeout(i, 1000 / 45);
            }; //各个浏览器支持的requestAnimationFrame有所不同，兼容各个浏览器

        function draw() {
            //清除画布
            context.clearRect(0, 0, W, H);

            var w = [mouse].concat(squares); //连接(合并)鼠标小方块数组和其他小方块数组
            var x, A;

            //square属性表：x，y，xa，ya，max
            squares.forEach(function (i) {
                //实现小方块定向移动
                i.x += i.xa;
                i.y += i.ya;

                // 控制小方块移动方向
                // 当小方块达到窗口边界时，反向移动
                i.xa = i.xa * (i.x > W || i.x < 0 ? -1 : 1);
                i.ya = i.ya * (i.y > H || i.y < 0 ? -1 : 1);

                //fillRect前两个参数为矩形左上角的x，y坐标，后两个分别为宽度和高度
                //绘制小方块
                context.fillRect(i.x - 0.5, i.y - 0.5, 1, 1);

                //遍历w中所有元素
                for (let n = 0; n < w.length; n++) {
                    x = w[n];

                    //如果x与i不是同一个对象实例且x的xy坐标存在
                    if (i !== x && null !== x.x && null !== x.y) {
                        let x_diff = i.x - x.x; //i和x的x坐标差

                        let y_diff = i.y - x.y; //i和x的y坐标差

                        let distance = x_diff * x_diff + y_diff * y_diff; //斜边平方

                        if (distance < x.max) {
                            //使i小方块受鼠标小方块束缚，即如果i小方块与鼠标小方块距离过大，i小方块会被鼠标小方块束缚,
                            //造成 多个小方块以鼠标为圆心，mouse.max/2为半径绕成一圈
                            if (x === mouse && distance > x.max / 2) {
                                i.x = i.x - 0.03 * x_diff;
                                i.y = i.y - 0.03 * y_diff;
                            }

                            A = (x.max - distance) / x.max;
                            context.beginPath();

                            //设置画笔的画线的粗细与两个小方块的距离相关，范围0-0.5，两个小方块距离越远画线越细，达到max时画线消失
                            context.lineWidth = A / 2;

                            //设置画笔的画线颜色为s.c即画布颜色，透明度为(A+0.2)即两个小方块距离越远画线越淡
                            context.strokeStyle = 'rgba(' + attr.color + ',' + (A + 0.2) + ')';

                            //设置画笔的笔触为i小方块
                            context.moveTo(i.x, i.y);

                            //使画笔的笔触移动到x小方块
                            context.lineTo(x.x, x.y);

                            //完成画线的绘制，即绘制连接小方块的线
                            context.stroke();
                        }
                    }
                }

                //把i小方块从w数组中去掉
                //防止两个小方块重复连线
                w.splice(w.indexOf(i), 1);
            });

            //window.requestAnimationFrame与setTimeout相似，形成递归调用，
            //不过window.requestAnimationFrame采用系统时间间隔，保持最佳绘制效率,提供了更好地优化，使动画更流畅
            //经过浏览器优化，动画更流畅；
            //窗口没激活时，动画将停止，省计算资源;
            animation(draw);
        }

        //此处是等待0.1秒后，执行一次draw()，真正的动画效果是用window.requestAnimationFrame实现的
        setTimeout(function () {
            draw();
        }, 100);
    }, []);

    return (
        <div style={{ backgroundColor: 'rgb(81, 102, 153)' }}>
            <canvas ref={canvasRef}></canvas>
            <div className="loginScreen">
                {/* <div className='logo'></div> */}
                <div className="title11">EUS-AI辅助诊断平台</div>
                <div
                    className={flag ? 'login-container' : 'login-container right-panel-active'}
                    id="container"
                >
                    <div className="form-container sign-up-container">
                        <div className="login-form" action="">
                            <h1 className="login-h1">创建账号</h1>
                            <span className="login-span">输入以注册用户名及密码</span>
                            <input
                                className="login-input"
                                type="text"
                                placeholder="用户名"
                                value={username || ''}
                                onChange={changeUsername}
                            ></input>
                            <input
                                className="login-input"
                                type="password"
                                placeholder="密码"
                                value={password || ''}
                                onChange={changePassword}
                            ></input>
                            <input
                                className="login-input"
                                type="password"
                                placeholder="请再输入一次密码"
                                value={passwordAgain || ''}
                                onChange={changePasswordAgain}
                            ></input>
                            <button className="login-button" onClick={signUp}>
                                注 册
                            </button>
                        </div>
                    </div>
                    <div className="form-container sign-in-container" style={{ color: 'black' }}>
                        <div className="login-form" action="">
                            <h2 className="login-h1 font-semibold" style={{ color: 'black' }}>
                                &nbsp;EUS-AI辅助诊断平台
                            </h2>
                            <h1 className="login-span font-light" style={{ color: 'black' }}>
                                Shanghai Sixth People's Hospital @ Department of Digestive Endoscopy
                            </h1>
                            <span className="login-span text-blue">DHU @ IIM</span>

                            <input
                                className="login-input"
                                type="text"
                                placeholder="用户名"
                                value={username || ''}
                                onChange={changeUsername}
                            />
                            <input
                                className="login-input"
                                type="password"
                                placeholder="密码"
                                value={password || ''}
                                onChange={changePassword}
                            />
                            <button className="login-button" onClick={signIn}>
                                登 入
                            </button>
                        </div>
                    </div>
                    <div className="overlay-container">
                        <div className="overlay">
                            <div className="overlay-panel overlay-left">
                                <h1 className="login-h1">欢 迎 回 来 !</h1>
                                <p className="login-p">如 果 已 有 账 号 请 在 这 登 入</p>
                                <button
                                    className="ghost login-button"
                                    id="signIn"
                                    onClick={changeClassName}
                                >
                                    登 入
                                </button>
                            </div>
                            <div className="overlay-panel overlay-right">
                                <h1 className="login-h1">欢迎注册使用</h1>
                                <p className="login-p">注 册 您 的 个 人 账 号 来 开 启 探 索</p>
                                <button
                                    className="ghost login-button"
                                    id="signUp"
                                    onClick={changeClassName}
                                >
                                    注 册
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
                <div className="bottom"></div>
            </div>
        </div>
    );
}
