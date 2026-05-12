import type { ProfileConfig } from "../types/config";

export const profileConfig: ProfileConfig = {
	// 头像
	// 图片路径支持三种格式：
	// 1. public 目录（以 "/" 开头，不优化）："/assets/images/avatar.webp"
	// 2. src 目录（不以 "/" 开头，自动优化但会增加构建时间，推荐）："assets/images/avatar.webp"
	// 3. 远程 URL："https://example.com/avatar.jpg"
	avatar: "/assets/images/tfpunch.GIF",

	// 名字
	name: "永雏多氢菲",

	// 个人签名
	bio: "∴さて····どこへ行こうか？",

	// 链接配置
	// 已经预装的图标集：fa7-brands，fa7-regular，fa7-solid，material-symbols，simple-icons
	// 访问https://icones.js.org/ 获取图标代码，
	// 如果想使用尚未包含相应的图标集，则需要安装它
	// `pnpm add @iconify-json/<icon-set-name>`
	// showName: true 时显示图标和名称，false 时只显示图标
	links: [
		{
			name: "Bilibili",
	         icon: "fa7-brands:bilibili",
	         url: "https://space.bilibili.com/22876427",
			showName: false,
		},
		{
			name: "Zhihu",
	         icon: "simple-icons:zhihu",
	         url: "https://www.zhihu.com/people/ranran-superposition",
			showName: false,
		},
		{
			name: "Book",
	         icon: "fa7-brands:readme",
	          url: "https://docs.qq.com/sheet/DRkZob0dkSGJGWERO?tab=BB08J2",
			showName: false,
		},
	],
};
