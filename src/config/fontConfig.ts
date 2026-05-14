// 字体配置
export const fontConfig = {
	// 是否启用自定义字体功能
	enable: true,
	// 是否预加载字体文件
	preload: true,
	// 当前选择的字体，支持多个字体组合
	selected: ["lxgw-wenkai", "gulliver-regular"],

	// 字体列表
	fonts: {
		// 系统字体
		system: {
			id: "system",
			name: "系统字体",
			src: "", // 系统字体无需 src
			family:
				"system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
		},

		// 霞鹜文楷 
		"lxgw-wenkai": {
			id: "lxgw-wenkai",
			name: "霞鹜文楷",
			src: "/assets/fonts/LXGWWenKai-Regular.ttf",
			family: "LXGW WenKai",
			display: "swap" as const,
		},

	// Gulliver Regular 英文专用（.otf 版本）
	"gulliver-regular": {
		id: "gulliver-regular",
		name: "Gulliver Regular",
		src: "/assets/fonts/Gulliver-Regular.otf",		
		family: "Gulliver",
		display: "swap" as const,
	},
	},

	// 全局字体回退
	fallback: [
	"Gulliver",        // 英文、数字优先
	"LXGW WenKai",     // 中文、日文优先
		"system-ui",
		"-apple-system",
		"BlinkMacSystemFont",
		"Segoe UI",
		"Roboto",
		"sans-serif",
	],
};