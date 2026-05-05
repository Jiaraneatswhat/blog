<script lang="ts">
	import type { CalendarGridCell } from "../types/calendar";

	interface Props {
		weekDays: string[];
		emptyCellsCount: number;
		cells: CalendarGridCell[];
		onCellClick: (dateKey: string) => void;
	}

	const { weekDays, emptyCellsCount, cells, onCellClick }: Props = $props();

	function getHeatStyle(cell: CalendarGridCell): string {
		if (cell.isEmpty) return "";

		// 选中态优先
		if (cell.isSelected) {
			return "background: #87b093;  color: white;";
		}

		// 今天优先
		if (cell.isToday) {
	return `
		background: #d5f6bc;
		color: #216e39;
		border: 1px solid #40c463;
	`;
}

		// 热力值（文章数）
		const count = cell.postCount;

		// 无数据：浅灰块
		if (count === 0) {
			return `
				background: rgba(156, 163, 175, 0.12);
				color: transparent;
			`;
		}
		if (count === 1) {
			return "background: #d5f6bc;";
		}
		if (count === 2) {
			return "background: #a4e0b4;";
		}
		if (count <= 4) {
			return "background: #95d1a6;";
		}
		if (count <= 6) {
			return "background:  #87b093; color: white;";
		}

		return "background:  #87b093; color: white;";
	}

	function getCellClass(cell: CalendarGridCell): string {
		if (cell.isEmpty) return "aspect-square";

		return `
			calendar-day
			group
			aspect-square
			flex
			items-center
			justify-center
			rounded-xl
			cursor-pointer
			relative
			transition-all
			duration-200
			border
			border-transparent
			text-neutral-700
			dark:text-neutral-300
			hover:scale-[1.05]
		`;
	}

	function handleCellClick(cell: CalendarGridCell) {
		if (!cell.isEmpty && cell.dateKey) {
			onCellClick(cell.dateKey);
		}
	}
</script>

<div class="grid grid-cols-7 gap-1 mb-2">
	{#each weekDays as day}
		<div
			class="text-center text-xs text-neutral-500 dark:text-neutral-400 font-medium py-1"
		>
			{day}
		</div>
	{/each}
</div>

<div class="grid grid-cols-7 gap-1">
	{#each { length: emptyCellsCount } as _}
		<div class="aspect-square"></div>
	{/each}

	{#each cells as cell (cell.dateKey)}
		{#if !cell.isEmpty}
			<button
				type="button"
				class={getCellClass(cell)}
				data-date={cell.dateKey}
				title={`${cell.day}日 · ${cell.postCount} 篇`}
				onclick={() => handleCellClick(cell)}
			>
				<div class="heat-block" style={getHeatStyle(cell)}>
				</div>
			</button>
		{/if}
	{/each}
</div>

<style>
	.heat-block {
		width: 90%;
		height: 90%;
		border-radius: 0.2rem;
		display: flex;
		align-items: center;
		justify-content: center;
		transition: all 0.2s ease;
	}
</style>