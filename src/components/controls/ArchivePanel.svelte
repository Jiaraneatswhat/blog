<script lang="ts">
import { onMount } from "svelte";

import I18nKey from "@/i18n/i18nKey";
import { i18n } from "@/i18n/translation";
import { getPostUrlBySlug } from "@/utils/url-utils";

export let tags: string[] = [];
export let categories: string[] = [];
export let sortedPosts: Post[] = [];

const params = new URLSearchParams(window.location.search);
tags = params.has("tag") ? params.getAll("tag") : [];
categories = params.has("category") ? params.getAll("category") : [];
const uncategorized = params.get("uncategorized");

interface Post {
	id: string;
	data: {
		title: string;
		tags: string[];
		category?: string | null;
		published: Date;
	};
}

interface Group {
	year: number;
	posts: Post[];
}

interface ActiveFilter {
	labelKey: I18nKey;
	values: string[];
}

let groups: Group[] = [];
let activeFilters: ActiveFilter[] = [];
let primaryFilter: ActiveFilter | null = null;
let secondaryFilters: ActiveFilter[] = [];
let filteredPostCount = 0;

function formatDate(date: Date) {
	const month = (date.getMonth() + 1).toString().padStart(2, "0");
	const day = date.getDate().toString().padStart(2, "0");
	return `${month}-${day}`;
}

function formatTag(tagList: string[]) {
	return tagList.map((t) => `#${t}`).join(" ");
}

function formatFilterValues(filter: ActiveFilter) {
	const prefix = filter.labelKey === I18nKey.tags ? "#" : "";
	return filter.values.map((value) => `${prefix}${value}`).join(" / ");
}

function resolvePrimaryFilter(filters: ActiveFilter[]) {
	return (
		filters.find((filter) => filter.labelKey === I18nKey.tags) ??
		filters[0] ??
		null
	);
}

function formatFilterSummary(filters: ActiveFilter[]) {
	return filters
		.map((filter) => `${i18n(filter.labelKey)}: ${formatFilterValues(filter)}`)
		.join("  ·  ");
}

// 🆕 根据标题和标签获取色块颜色
function getCategoryColor(post: Post): string {
	const title = post.data.title || '';
	const tags = post.data.tags?.map(t => t.toLowerCase()) || [];

	// 芥川龍之介，江戸川乱歩，梶井基次郎，中島敦，夏目漱石 → #16ad6b
	if (
		title.includes('芥川龍之介') ||
		title.includes('江戸川乱歩') ||
		title.includes('梶井基次郎') ||
		title.includes('中島敦') ||
		title.includes('夏目漱石')
	) {
		return 'bg-[#16ad6b]';
	}

	// 鎌池和馬，橘公司，西尾維新 → #dfd7d4
	if (
		title.includes('鎌池和馬') ||
		title.includes('橘公司') ||
		title.includes('西尾維新')
	) {
		return 'bg-[#dfd7d4]';
	}

	// 村上春樹，谷崎潤一郎，坂口安吾，白井智之，背筋 → #2e8b57
	if (
		title.includes('村上春樹') ||
		title.includes('谷崎潤一郎') ||
		title.includes('坂口安吾') ||
		title.includes('白井智之') ||
		title.includes('背筋')
	) {
		return 'bg-[#2e8b57]';
	}

	// 生物学 → #f8e411（标签判断）
	if (tags.includes('生物学')) {
		return 'bg-[#f8e411]';
	}

	// 期刊 → #02aedb（标签判断）
	if (
		tags.includes('期刊') ||
		tags.includes('journal')
	) {
		return 'bg-[#02aedb]';
	}

	// 轻小说 → #dfd7d4（标签判断）
	if (tags.includes('轻小说')) {
		return 'bg-[#dfd7d4]';
	}

	// 其他 → 灰色
	return 'bg-gray-400';
}

onMount(async () => {
	let filteredPosts: Post[] = sortedPosts;
	const currentFilters: ActiveFilter[] = [];

	if (categories.length > 0) {
		currentFilters.push({ labelKey: I18nKey.categories, values: categories });
	}

	if (uncategorized) {
		currentFilters.push({
			labelKey: I18nKey.categories,
			values: [i18n(I18nKey.uncategorized)],
		});
	}

	if (tags.length > 0) {
		currentFilters.push({ labelKey: I18nKey.tags, values: tags });
	}

	activeFilters = currentFilters;
	primaryFilter = resolvePrimaryFilter(activeFilters);
	secondaryFilters = primaryFilter
		? activeFilters.filter((filter) => filter !== primaryFilter)
		: [];

	if (tags.length > 0) {
		filteredPosts = filteredPosts.filter(
			(post) =>
				Array.isArray(post.data.tags) &&
				post.data.tags.some((tag) => tags.includes(tag)),
		);
	}

	if (categories.length > 0) {
		filteredPosts = filteredPosts.filter(
			(post) => post.data.category && categories.includes(post.data.category),
		);
	}

	if (uncategorized) {
		filteredPosts = filteredPosts.filter((post) => !post.data.category);
	}

	filteredPosts = filteredPosts
		.slice()
		.sort((a, b) => b.data.published.getTime() - a.data.published.getTime());

	filteredPostCount = filteredPosts.length;

	const grouped = filteredPosts.reduce(
		(acc, post) => {
			const year = post.data.published.getFullYear();
			if (!acc[year]) {
				acc[year] = [];
			}
			acc[year].push(post);
			return acc;
		},
		{} as Record<number, Post[]>,
	);

	const groupedPostsArray = Object.keys(grouped).map((yearStr) => ({
		year: Number.parseInt(yearStr, 10),
		posts: grouped[Number.parseInt(yearStr, 10)],
	}));

	groupedPostsArray.sort((a, b) => b.year - a.year);

	groups = groupedPostsArray;
});
</script>

<div class="card-base px-8 py-6">
	{#if primaryFilter}
		<div class="mb-5">
			<div class="flex flex-wrap items-center justify-between gap-x-4 gap-y-1">
				<div class="min-w-0 text-sm text-75">
					<span class="text-50">{i18n(primaryFilter.labelKey)}</span>
					<span class="mx-2 text-30">/</span>
					<span class="font-semibold text-(--primary)">{formatFilterValues(primaryFilter)}</span>
					{#if secondaryFilters.length > 0}
						<span class="ml-2 text-50">· {formatFilterSummary(secondaryFilters)}</span>
					{/if}
				</div>
				<div class="shrink-0 text-xs text-50">
					{filteredPostCount} {i18n(filteredPostCount === 1 ? I18nKey.postCount : I18nKey.postsCount)}
					<span class="mx-1.5 text-30">·</span>
					{groups.length} {i18n(I18nKey.year)}
				</div>
			</div>
		</div>
	{/if}

	{#each groups as group}
		<div>
			<div class="flex flex-row w-full items-center h-15">
				<div class="w-[15%] md:w-[10%] transition text-2xl font-bold text-right text-75">
					{group.year}
				</div>
				<div class="w-[15%] md:w-[10%]">
					<div
						class="h-3 w-3 bg-none rounded-full outline outline-(--primary) mx-auto
                  -outline-offset-2 z-50 outline-3"
					></div>
				</div>
				<div class="w-[70%] md:w-[80%] transition text-left text-50">
					{group.posts.length} {i18n(group.posts.length === 1 ? I18nKey.postCount : I18nKey.postsCount)}
				</div>
			</div>

			{#each group.posts as post}
				<a
					href={getPostUrlBySlug(post.id)}
					aria-label={post.data.title}
					class="group btn-plain block! h-10 w-full rounded-lg hover:text-[initial]"
				>
					<div class="flex flex-row justify-start items-center h-full">
						<!-- date -->
						<div class="w-[15%] md:w-[10%] transition text-sm text-right text-50">
							{formatDate(post.data.published)}
						</div>

						<!-- dot and line -->
						<div class="w-[15%] md:w-[10%] relative dash-line h-full flex items-center">
							<div
								class="transition-all mx-auto w-1 h-1 rounded group-hover:h-5
                       bg-[oklch(0.5_0.05_var(--hue))] group-hover:bg-(--primary)
                       outline outline-4 z-50
                       outline-(--card-bg)
                       group-hover:outline-(--btn-plain-bg-hover)
                       group-active:outline-(--btn-plain-bg-active)"
							></div>
						</div>

						<!-- post title with color block -->
<div
    class="w-[70%] md:max-w-[65%] md:w-[65%] text-left font-bold
         group-hover:translate-x-1 transition-all group-hover:text-(--primary)
         text-75 pr-8 flex items-center gap-2.5"
>
    <span 
        class="flex-shrink-0 w-4 h-4 rounded-md {getCategoryColor(post)}"
        title={post.data.tags[0] || post.data.category || '未分类'}
    ></span>
    <span class="whitespace-nowrap text-ellipsis overflow-hidden">
        {post.data.title}
    </span>
</div>

						<!-- tag list -->
						<div
							class="hidden md:block md:w-[15%] text-left text-sm transition
                     whitespace-nowrap text-ellipsis overflow-hidden text-30"
						>
							{formatTag(post.data.tags)}
						</div>
					</div>
				</a>
			{/each}
		</div>
	{/each}
</div>