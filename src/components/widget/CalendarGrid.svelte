<script lang="ts">
	import { onMount } from "svelte";

	interface PostMeta {
		id: string;
		title: string;
		published: string;
	}

	interface CalendarCell {
		date: string;
		dateKey: string;
		day: number;
		count: number;
		level: number;
		isToday: boolean;
		isSelected: boolean;
		posts: PostMeta[];
	}

	let allPostsData: PostMeta[] = [];
	let cells: CalendarCell[] = [];
	let selectedDateKey = "";
	let selectedPosts: PostMeta[] = [];
	let displayYear = new Date().getFullYear();
	let displayMonth = new Date().getMonth();
	let weekDays = ["日", "一", "二", "三", "四", "五", "六"];

	const monthNames = ["1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月"];

	onMount(async () => {
		await fetchData();
		buildCalendar();
	});

	async function fetchData() {
		try {
			const response = await fetch("/api/allPostMeta.json");
			allPostsData = await response.json();
		} catch (error) {
			console.error("Failed to fetch calendar data", error);
		}
	}

	function getPostCount(dateStr: string): { count: number; posts: PostMeta[] } {
		const posts = allPostsData.filter((post) => {
			const d = new Date(post.published);
			const key = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
			return key === dateStr;
		});
		return { count: posts.length, posts };
	}

	function getHeatLevel(count: number): number {
		if (count === 0) return 0;
		if (count === 1) return 1;
		if (count === 2) return 2;
		if (count <= 4) return 3;
		return 4;
	}

	function buildCalendar() {
		cells = [];
		selectedPosts = [];
		selectedDateKey = "";

		const today = new Date();
		const todayKey = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, "0")}-${String(today.getDate()).padStart(2, "0")}`;

		const firstDayOfMonth = new Date(displayYear, displayMonth, 1).getDay();
		const daysInMonth = new Date(displayYear, displayMonth + 1, 0).getDate();

		// 月初空白
		for (let i = 0; i < firstDayOfMonth; i++) {
			cells.push({
				date: "",
				dateKey: "",
				day: 0,
				count: 0,
				level: -1,
				isToday: false,
				isSelected: false,
				posts: [],
			});
		}

		// 每一天
		for (let d = 1; d <= daysInMonth; d++) {
			const dateKey = `${displayYear}-${String(displayMonth + 1).padStart(2, "0")}-${String(d).padStart(2, "0")}`;
			const { count, posts } = getPostCount(dateKey);
			cells.push({
				date: `${displayMonth + 1}/${d}`,
				dateKey,
				day: d,
				count,
				level: getHeatLevel(count),
				isToday: dateKey === todayKey,
				isSelected: dateKey === selectedDateKey,
				posts,
			});
		}

		// 显示当月所有文章
		const currentMonthPosts = allPostsData.filter((post) => {
			const date = new Date(post.published);
			return date.getFullYear() === displayYear && date.getMonth() === displayMonth;
		});
		selectedPosts = currentMonthPosts;
	}

	function getHeatColor(level: number): string {
		const colors = [
			"bg-neutral-200 dark:bg-neutral-700",
			"bg-emerald-200 dark:bg-emerald-800",
			"bg-emerald-300 dark:bg-emerald-700",
			"bg-emerald-400 dark:bg-emerald-600",
			"bg-emerald-500 dark:bg-emerald-500",
		];
		return level >= 0 ? colors[level] : "bg-transparent";
	}

	function handleCellClick(cell: CalendarCell) {
		if (cell.level < 0) return;

		cells = cells.map((c) => ({ ...c, isSelected: false }));
		cell.isSelected = true;
		selectedDateKey = cell.dateKey;
		selectedPosts = cell.posts;
		cells = [...cells];
	}

	function changeMonth(delta: number) {
		displayMonth += delta;
		if (displayMonth > 11) {
			displayMonth = 0;
			displayYear++;
		} else if (displayMonth < 0) {
			displayMonth = 11;
			displayYear--;
		}
		buildCalendar();
	}

	function resetToToday() {
		const now = new Date();
		displayYear = now.getFullYear();
		displayMonth = now.getMonth();
		buildCalendar();
	}
</script>

<div class="calendar-github-wrapper">
	<!-- 头部：月份切换 -->
	<div class="flex items-center justify-between mb-3">
		<button
			class="btn-plain rounded-lg w-8 h-8 flex items-center justify-center hover:bg-(--btn-plain-bg-hover) transition-colors"
			on:click={() => changeMonth(-1)}
			aria-label="上个月"
		>
			<svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 18 9 12 15 6"></polyline></svg>
		</button>

		<span class="text-base font-bold text-neutral-900 dark:text-neutral-100">
			{displayYear}年{monthNames[displayMonth]}
		</span>

		<div class="flex gap-2">
			<button
				class="btn-plain rounded-lg w-8 h-8 flex items-center justify-center hover:bg-(--btn-plain-bg-hover) transition-colors"
				on:click={resetToToday}
				aria-label="回到今天"
			>
				<svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"></path><path d="M3 3v5h5"></path></svg>
			</button>
			<button
				class="btn-plain rounded-lg w-8 h-8 flex items-center justify-center hover:bg-(--btn-plain-bg-hover) transition-colors"
				on:click={() => changeMonth(1)}
				aria-label="下个月"
			>
				<svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"></polyline></svg>
			</button>
		</div>
	</div>

	<!-- 图例 -->
	<div class="flex items-center justify-end gap-1 mb-3 text-xs text-neutral-500 dark:text-neutral-400">
		<span>Less</span>
		<div class="w-3 h-3 rounded-sm bg-neutral-200 dark:bg-neutral-700"></div>
		<div class="w-3 h-3 rounded-sm bg-emerald-200 dark:bg-emerald-800"></div>
		<div class="w-3 h-3 rounded-sm bg-emerald-300 dark:bg-emerald-700"></div>
		<div class="w-3 h-3 rounded-sm bg-emerald-400 dark:bg-emerald-600"></div>
		<div class="w-3 h-3 rounded-sm bg-emerald-500 dark:bg-emerald-500"></div>
		<span>More</span>
	</div>

	<!-- 星期头 -->
	<div class="grid grid-cols-7 gap-1 mb-1">
		{#each weekDays as day}
			<div class="text-center text-xs text-neutral-500 dark:text-neutral-400 font-medium py-1">
				{day}
			</div>
		{/each}
	</div>

	<!-- 日期热力图网格 -->
	<div class="grid grid-cols-7 gap-1">
		{#each cells as cell}
			{#if cell.level >= 0}
				<button
					type="button"
					class="aspect-square rounded-md flex items-center justify-center text-sm font-bold transition-all hover:scale-110
						{getHeatColor(cell.level)}
						{cell.level >= 3 ? 'text-white' : 'text-neutral-700 dark:text-neutral-300'}
						{cell.isToday ? 'ring-2 ring-(--primary) ring-offset-1 dark:ring-offset-neutral-900' : ''}
						{cell.isSelected ? 'ring-1 ring-neutral-400' : ''}"
					title="{cell.date} · {cell.count} 篇"
					on:click={() => handleCellClick(cell)}
				>
					{cell.day}
				</button>
			{:else}
				<div class="aspect-square"></div>
			{/if}
		{/each}
	</div>

	<!-- 下方文章列表 -->
	{#if selectedPosts.length > 0}
		<div class="mt-3 pt-3 border-t border-neutral-200 dark:border-neutral-700">
			<div class="text-xs text-neutral-500 dark:text-neutral-400 mb-2">
				{selectedDateKey ? `${selectedDateKey} · ${selectedPosts.length} 篇` : `${displayYear}年${monthNames[displayMonth]} · ${selectedPosts.length} 篇`}
			</div>
			<div class="flex flex-col gap-1 max-h-40 overflow-y-auto">
				{#each selectedPosts as post}
					<a
						href="/posts/{post.id}/"
						class="flex justify-between items-center text-sm text-neutral-700 dark:text-neutral-300 hover:text-(--primary) transition-colors px-2 py-1 rounded hover:bg-(--btn-plain-bg-hover)"
					>
						<span class="truncate">{post.title}</span>
						<span class="text-xs text-neutral-500 dark:text-neutral-400 ml-2 whitespace-nowrap">
							{new Date(post.published).getMonth() + 1}/{new Date(post.published).getDate()}
						</span>
					</a>
				{/each}
			</div>
		</div>
	{/if}
</div>