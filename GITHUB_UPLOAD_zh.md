# 上传到 GitHub：按这份操作即可

## 1. 解压并找到要上传的文件

解压 `nmr-archie-symbolic-regression-v0.1.0.zip`，打开里面的
`nmr-archie-symbolic-regression` 文件夹。

只上传这个文件夹里面的内容。根目录应包含：

```text
README.md
LICENSE
pyproject.toml
requirements.txt
.gitignore
GITHUB_UPLOAD_zh.md
src/
data/
examples/
scripts/
tests/
docs/
```

包中已经放入修改后的代码、80 个模拟岩心的 400 条观测、实际运行结果、
模型文件、测试及英文说明。没有放入你的保密数据，也没有放入论文原稿。
公开许可采用 MIT，适用于代码与随包模拟材料。

## 2. 在网页上创建公开仓库

1. 登录 [GitHub](https://github.com/)。
2. 点击右上角 `+` → `New repository`，或打开 [新建仓库页面](https://github.com/new)。
3. `Repository name` 填：`nmr-archie-symbolic-regression`。
4. `Description` 可以填：
   `NMR-constrained Archie-type symbolic regression with synthetic data and reproducible examples.`
5. 可见性选择 **Public**，这样编辑和审稿人可以直接访问。
6. 创建时不用自动添加 README、.gitignore 或许可证，因为包内已有。
7. 点击 `Create repository`。

## 3. 上传解压后的内容

1. 空仓库页面点击 `uploading an existing file`。
   已有文件的仓库使用 `Add file` → `Upload files`。
2. 把第 1 步列出的文件和文件夹拖到上传区域。
   **上传解压后的内容，不能只上传 ZIP。**
3. 等待上传结束。检查 `README.md` 在仓库根目录；
   `src/nmr_archie_sr/` 和 `data/synthetic_cores.csv` 的目录层次要保留。
4. 提交说明填：`Initial release: corrected implementation and synthetic examples`。
5. 提交到默认分支，点击界面中的 `Commit changes` 或对应提交按钮。
   如果界面要求创建分支/提出更改，按页面指示完成提交。

网页一次最多上传 100 个文件，每个文件不超过 25 MiB；此包在这个范围内。
拖动文件夹时，Chrome/Edge 通常更方便。若看不到 `.gitignore`，在资源管理器
启用隐藏文件显示，或在仓库用 `Add file` → `Create new file` 单独添加。

## 4. 上传后检查四处

1. 仓库首页自动展示英文 README，`LICENSE` 显示为 MIT。
2. `data/synthetic_cores.csv` 可直接打开，编号全部为 `SYNTHETIC_...`。
3. `examples/results/double_q/` 中可查看模型、指标和预测结果。
4. 复制仓库地址，在浏览器无痕窗口打开，确认不登录也能看到文件。

仓库地址的形式是 `https://github.com/你的用户名/nmr-archie-symbolic-regression`。
这只是地址格式；论文里必须填写你创建后实际存在的完整链接。

可再创建 `v0.1.0` Release：仓库 `Releases` → `Draft a new release`，
标签填 `v0.1.0`，目标选择默认分支，说明填
`Implementation and synthetic examples for reproducible computational workflows.`
然后发布。这样后续改代码时仍能指向本次固定版本。

## 5. 论文中代码和数据声明怎么写

**以下是草稿，先把方括号内容替换为真实仓库/版本链接，再放入论文。**

> The code is publicly available at [INSERT VERIFIED REPOSITORY URL] under the MIT
> license. Because the experimental core and well-log data are subject to
> confidentiality restrictions, synthetic example data are provided for code
> execution, testing and reproduction of the computational workflow.

复现说明应准确记录所用数据、配置和软件环境，并区分计算流程复现与具体实验
数值的核验。

## 6. 本次修改与投稿之间还差什么

本代码包已提供合成示例运行记录。提交作者已确认本研究使用该代码，以下是
投稿时应核对的方法与材料事项：

- 核对论文公式、方法、表格、图和运行配置中的参数、单位与评价范围。
- 当前 n 的拟合使用实测 R0，属于独立标定；如果论文写成 m 固定后联合反演，
  需要与真实实现统一。
- NMR-MICP 预处理、测井验证及 PySR/gplearn 对比代码不在本次收到的源码中，
  这个仓库不声称覆盖这些环节。
- 在 README 中补充最终论文题名和经核对的作者名单；当前没有编造文章引用或 DOI。
- 本次小规模模拟对照未显示 Double Q 优于纯 MCTS，不能据此新增算法优势结论。

以后若用本项目跑保密数据，请放在 `private_data/`，输出放在 `outputs/`。
这两处已加入 `.gitignore`，但**网页拖动上传不会遵守 .gitignore**；
只上传本次整理好的公开包，不要拖入整个工作区或以后生成的私人结果。

## 官方说明

- [GitHub 创建仓库](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository)
- [GitHub 网页上传文件](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository)
- [Computers & Geosciences 期刊主页与作者说明](https://shop.elsevier.com/journals/computers-and-geosciences/0098-3004)
- [C&G 代码仓库说明](https://github.com/cageo/author_instructions)

代码公开只是投稿材料的一部分。是否符合期刊范围还取决于论文的计算方法贡献
与地学问题，整理代码包本身不能保证解决此前的范围问题。
