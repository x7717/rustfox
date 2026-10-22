# rustfox

Firefox-подход к оптимизации браузерных hot-path на Rust: цикл
**бенчмарк → профилирование → оптимизация → бенчмарк**, крутящийся локально и в CI.

## Что внутри

| крейт | назначение |
|---|---|
| `rustfox-dom` | arena-DOM с интернированием строк (теги/классы → `u32`) — сравнения в hot path целочисленные, как atom table в Stylo |
| `rustfox-style` | парсер CSS-селекторов + три матчера: `naive` (база), `optimized` (bloom-фильтр предков), `indexed` (rule bucketing + bloom + flat zero-alloc выход) |

Текущий результат: **4.4–5.3x** к наивной базе на style-matching проходе
при байт-в-байт идентичных результатах (parity-тесты). Подробности и история
итераций с числами — [WORKLOG.md](WORKLOG.md), свежие цифры — [bench/latest.json](bench/latest.json).

## Быстрый старт

```bash
cargo test          # корректность: naive == optimized == indexed на фикстурах
cargo bench         # criterion: naive vs bloom vs indexed
```

## CI

| workflow | что делает | расписание |
|---|---|---|
| `ci.yml` | fmt, clippy `-D warnings`, тесты, компиляция бенчей | на каждый push/PR |
| `bench.yml` | criterion-прогон, результаты коммитятся в ветку `bench-results` (трекинг дрейфа производительности) | cron `0 3 * * *` + вручную |
| `profile.yml` | perf + flamegraph SVG артефактом — искать следующее горячее место | cron `0 4 * * 1` + вручную |

Смежный репозиторий: [x7717/gecko-dev](https://github.com/x7717/gecko-dev) — форк
официального зеркала Firefox с CI для Rust-компонентов (WebRender) и планом
оптимизаций в [FIREFOX_RUST_PLAN.md](https://github.com/x7717/gecko-dev/blob/master/FIREFOX_RUST_PLAN.md).

## Честные рамки проекта

Полный «форк Firefox на Rust» — это проект масштаба Servo (годы работы большой
команды); Gecko уже содержит большие Rust-компоненты (Stylo, WebRender, neqo).
Этот репозиторий делает то, что реально и измеримо: воспроизводит техники
production-движков (rule hash, ancestor bloom filter, интернинг, flat-структуры)
на настоящем Rust-коде, гоняет их через автоматизированный цикл оптимизаций
с числами до/после на каждом шаге и переносит проверенные техники в апстрим-план.
