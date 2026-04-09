import io
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image


# =========================
# 页面基础设置
# =========================
st.set_page_config(
    page_title="图像识别课程作业系统",
    page_icon="📘",
    layout="wide"
)

st.title("图像识别课程作业演示系统")
st.markdown(
    """
本系统整合了两部分内容：

1. **Least Squares Linear Regression（最小二乘线性回归）**
2. **KNN 图像分类（支持 zip.train 与自定义图片）**

左侧可切换功能模块。
"""
)


# =========================
# 通用工具函数
# =========================
def safe_load_txt(file_obj_or_path):
    """读取文本数据。既支持本地路径，也支持 Streamlit 上传文件。"""
    return np.loadtxt(file_obj_or_path)


def normalize_to_minus1_1(img_array):
    """将 0~255 像素归一化到 [-1, 1]。"""
    return img_array / 127.5 - 1.0


def image_to_16x16_vector(uploaded_file_or_pil, invert=False):
    """
    把上传图片处理成与 zip.train 兼容的 16x16 灰度向量（256维）
    返回：
        vec: shape=(256,)
        img16: shape=(16,16)
        img_show: PIL.Image
    """
    if isinstance(uploaded_file_or_pil, Image.Image):
        img = uploaded_file_or_pil.convert("L")
    else:
        img = Image.open(uploaded_file_or_pil).convert("L")

    img = img.resize((16, 16))
    img_arr = np.array(img, dtype=np.float32)

    if invert:
        img_arr = 255.0 - img_arr

    vec = normalize_to_minus1_1(img_arr).flatten()
    return vec, img_arr, img


# =========================
# 第一题：最小二乘线性回归
# =========================
def parse_xy_text(x_text, y_text):
    """
    把用户输入的逗号分隔字符串解析成 numpy 数组
    例如:
    x_text = "1,2,3,4"
    y_text = "2,4,5,8"
    """
    x = np.array([float(i.strip()) for i in x_text.split(",") if i.strip() != ""])
    y = np.array([float(i.strip()) for i in y_text.split(",") if i.strip() != ""])
    return x, y


def least_squares_fit(x, y):
    """
    一元线性回归闭式解:
    y = w*x + b
    theta = (X^T X)^(-1) X^T y
    """
    x_hat = np.c_[np.ones(len(x)), x]
    xtx = x_hat.T @ x_hat

    # 为了更稳妥，优先尝试 solve；失败则退到伪逆
    try:
        theta = np.linalg.solve(xtx, x_hat.T @ y)
    except np.linalg.LinAlgError:
        theta = np.linalg.pinv(xtx) @ x_hat.T @ y

    b = float(theta[0])
    w = float(theta[1])
    return w, b


def predict_linear(x, w, b):
    return w * x + b


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def make_regression_plot(x, y, w, b):
    y_pred = predict_linear(x, w, b)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(x, y)
    x_line = np.linspace(np.min(x), np.max(x), 200)
    ax.plot(x_line, predict_linear(x_line, w, b))
    ax.set_title("Least Squares Linear Regression")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.grid(True)
    return fig, y_pred


def regression_demo_data():
    np.random.seed(42)
    x = np.linspace(0, 10, 30)
    noise = np.random.randn(30) * 2
    y = 3 * x + 5 + noise
    return x, y


# =========================
# 第二题：KNN 图像分类
# =========================
@st.cache_data
def load_zip_train_dataset(file_bytes):
    """
    读取 zip.train 文件内容（bytes），返回 X, y
    每行格式:
    label pixel1 pixel2 ... pixel256
    """
    data = np.loadtxt(io.BytesIO(file_bytes))
    y = data[:, 0].astype(int)
    X = data[:, 1:].astype(np.float32)
    return X, y


def train_test_split_manual(X, y, test_size=0.2, random_state=42):
    rng = np.random.default_rng(random_state)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    test_count = int(len(X) * test_size)

    test_idx = idx[:test_count]
    train_idx = idx[test_count:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def euclidean_distance_batch(X_train, x_test):
    """向量化计算单个测试样本到全部训练样本的欧氏距离。"""
    return np.sqrt(np.sum((X_train - x_test) ** 2, axis=1))


def knn_predict_single(X_train, y_train, x_test, k=3):
    distances = euclidean_distance_batch(X_train, x_test)
    nearest_idx = np.argsort(distances)[:k]
    nearest_labels = y_train[nearest_idx]

    values, counts = np.unique(nearest_labels, return_counts=True)
    pred = int(values[np.argmax(counts)])

    neighbors = []
    for idx in nearest_idx:
        neighbors.append({
            "index": int(idx),
            "label": int(y_train[idx]),
            "distance": float(distances[idx])
        })
    return pred, neighbors


def knn_predict_batch(X_train, y_train, X_test, k=3):
    preds = []
    for x in X_test:
        pred, _ = knn_predict_single(X_train, y_train, x, k=k)
        preds.append(pred)
    return np.array(preds, dtype=int)


def accuracy_score(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def confusion_matrix_manual(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(cm.shape[1]))
    ax.set_yticks(range(cm.shape[0]))

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax)
    return fig


def plot_sample_predictions(X, y_true, y_pred, sample_count=9):
    sample_count = min(sample_count, len(X))
    rng = np.random.default_rng(123)
    idxs = rng.choice(len(X), size=sample_count, replace=False)

    rows = math.ceil(sample_count / 3)
    fig, axes = plt.subplots(rows, 3, figsize=(9, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for i, idx in enumerate(idxs):
        ax = axes[i]
        ax.imshow(X[idx].reshape(16, 16))
        ax.set_title(f"True={y_true[idx]} / Pred={y_pred[idx]}")
        ax.axis("off")

    plt.tight_layout()
    return fig


def plot_neighbors(input_img16, pred_label, X_train, neighbors):
    cols = len(neighbors) + 1
    fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3))

    axes[0].imshow(input_img16)
    axes[0].set_title(f"Input\nPred={pred_label}")
    axes[0].axis("off")

    for i, n in enumerate(neighbors, start=1):
        idx = n["index"]
        label = n["label"]
        dist = n["distance"]
        axes[i].imshow(X_train[idx].reshape(16, 16))
        axes[i].set_title(f"Label={label}\nDist={dist:.2f}")
        axes[i].axis("off")

    plt.tight_layout()
    return fig


def try_predict_uploaded_image(X_train, y_train, uploaded_img, k):
    """
    同时尝试：
    1. 原图
    2. 反色图
    选择最近邻平均距离更小的结果
    """
    vec1, img16_1, _ = image_to_16x16_vector(uploaded_img, invert=False)
    pred1, neigh1 = knn_predict_single(X_train, y_train, vec1, k=k)
    avg1 = np.mean([n["distance"] for n in neigh1])

    uploaded_img.seek(0)
    vec2, img16_2, _ = image_to_16x16_vector(uploaded_img, invert=True)
    pred2, neigh2 = knn_predict_single(X_train, y_train, vec2, k=k)
    avg2 = np.mean([n["distance"] for n in neigh2])

    if avg1 <= avg2:
        return {
            "pred": pred1,
            "neighbors": neigh1,
            "img16": img16_1,
            "mode": "原图"
        }
    else:
        return {
            "pred": pred2,
            "neighbors": neigh2,
            "img16": img16_2,
            "mode": "反色图"
        }


# =========================
# 侧边栏
# =========================
module = st.sidebar.selectbox(
    "选择功能模块",
    ["首页", "第一题：Least Squares Linear Regression", "第二题：KNN 图像分类", "关于部署"]
)

st.sidebar.markdown("---")
st.sidebar.write("可在 GitHub + Streamlit 部署")


# =========================
# 首页
# =========================
if module == "首页":
    st.subheader("功能概览")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
### 第一题
- 自定义输入 X、y 数据
- 最小二乘法闭式解
- 输出参数 w、b、MSE
- 绘制拟合直线
"""
        )

    with col2:
        st.markdown(
            """
### 第二题
- 上传 `zip.train`
- 设置 `k` 值
- 测试集准确率与混淆矩阵
- 上传单张图片并预测
"""
        )

    st.info("建议先在左侧切到对应模块运行。")


# =========================
# 第一题模块
# =========================
elif module == "第一题：Least Squares Linear Regression":
    st.subheader("第一题：Least Squares Linear Regression")

    use_demo = st.checkbox("使用示例数据", value=True)

    if use_demo:
        x_demo, y_demo = regression_demo_data()
        default_x = ",".join([f"{v:.2f}" for v in x_demo])
        default_y = ",".join([f"{v:.2f}" for v in y_demo])
    else:
        default_x = "1,2,3,4,5"
        default_y = "2,4,5,4,6"

    with st.form("regression_form"):
        x_text = st.text_area("请输入 X（逗号分隔）", value=default_x, height=120)
        y_text = st.text_area("请输入 y（逗号分隔）", value=default_y, height=120)
        submitted = st.form_submit_button("开始拟合")

    if submitted:
        try:
            x, y = parse_xy_text(x_text, y_text)

            if len(x) != len(y):
                st.error("X 和 y 的长度必须相同。")
            elif len(x) < 2:
                st.error("至少需要两个样本点。")
            else:
                w, b = least_squares_fit(x, y)
                fig, y_pred = make_regression_plot(x, y, w, b)
                err = mse(y, y_pred)

                c1, c2, c3 = st.columns(3)
                c1.metric("w", f"{w:.6f}")
                c2.metric("b", f"{b:.6f}")
                c3.metric("MSE", f"{err:.6f}")

                st.write(f"回归方程：**y = {w:.6f}x + {b:.6f}**")
                st.pyplot(fig)

                result_df = pd.DataFrame({
                    "X": x,
                    "y_true": y,
                    "y_pred": y_pred
                })
                st.dataframe(result_df, use_container_width=True)

        except Exception as e:
            st.error(f"输入解析失败：{e}")


# =========================
# 第二题模块
# =========================
elif module == "第二题：KNN 图像分类":
    st.subheader("第二题：KNN 图像分类")

    st.markdown(
        """
请先上传 `zip.train`。该文件每行应是：

`label pixel1 pixel2 ... pixel256`

其中前 1 列是标签，后 256 列对应 16×16 图像。
"""
    )

    zip_train_file = st.file_uploader(
        "上传 zip.train 文件",
        type=["train", "txt", "data", "csv"]
    )

    k = st.number_input("选择 K 值", min_value=1, max_value=15, value=3, step=1)
    test_ratio = st.slider("测试集比例", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    random_state = st.number_input("随机种子", min_value=0, max_value=9999, value=42, step=1)

    if zip_train_file is not None:
        try:
            file_bytes = zip_train_file.getvalue()
            X, y = load_zip_train_dataset(file_bytes)

            st.success("数据集读取成功。")

            c1, c2, c3 = st.columns(3)
            c1.metric("样本数", str(len(X)))
            c2.metric("特征维度", str(X.shape[1]))
            c3.metric("类别数", str(len(np.unique(y))))

            dist_df = pd.Series(y).value_counts().sort_index().rename_axis("label").reset_index(name="count")
            st.write("类别分布：")
            st.dataframe(dist_df, use_container_width=True)

            with st.form("knn_eval_form"):
                run_eval = st.form_submit_button("运行 KNN 测试集评估")

            if run_eval:
                X_train, X_test, y_train, y_test = train_test_split_manual(
                    X, y, test_size=float(test_ratio), random_state=int(random_state)
                )

                y_pred = knn_predict_batch(X_train, y_train, X_test, k=int(k))
                acc = accuracy_score(y_test, y_pred)
                cm = confusion_matrix_manual(y_test, y_pred, num_classes=10)

                st.metric("测试集准确率", f"{acc * 100:.2f}%")

                fig_samples = plot_sample_predictions(X_test, y_test, y_pred, sample_count=9)
                st.pyplot(fig_samples)

                fig_cm = plot_confusion_matrix(cm)
                st.pyplot(fig_cm)

                result_df = pd.DataFrame({
                    "y_true": y_test,
                    "y_pred": y_pred
                })
                st.dataframe(result_df.head(50), use_container_width=True)

            st.markdown("---")
            st.subheader("上传单张图片进行预测")

            uploaded_img = st.file_uploader(
                "上传一张数字图片（jpg/png/jpeg）",
                type=["jpg", "jpeg", "png"],
                key="image_predict"
            )

            if uploaded_img is not None:
                # 重新划分训练集，和前面评估保持一致
                X_train, X_test, y_train, y_test = train_test_split_manual(
                    X, y, test_size=float(test_ratio), random_state=int(random_state)
                )

                left, right = st.columns([1, 1])

                with left:
                    st.image(uploaded_img, caption="上传的原始图片", use_container_width=True)

                uploaded_img.seek(0)
                result = try_predict_uploaded_image(X_train, y_train, uploaded_img, int(k))

                with right:
                    st.success(f"预测结果：{result['pred']}")
                    st.write(f"采用模式：{result['mode']}")

                    neighbor_df = pd.DataFrame(result["neighbors"])
                    st.dataframe(neighbor_df, use_container_width=True)

                fig_neighbors = plot_neighbors(
                    result["img16"],
                    result["pred"],
                    X_train,
                    result["neighbors"]
                )
                st.pyplot(fig_neighbors)

                st.caption("提示：如果图片是黑底白字或白底黑字不确定，程序会自动比较原图与反色图。")

        except Exception as e:
            st.error(f"数据读取或计算失败：{e}")
    else:
        st.info("请先上传 zip.train 文件。")


