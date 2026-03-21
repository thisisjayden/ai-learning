# 🔥 Top 20 GitHub Copilot Skills & Extensions (Deep Dive)

*Compiled based on highest adoption, GitHub Marketplace rankings, and developer workflows.*

## 🧩 Part 1: Top 10 Marketplace Extensions (Ecosystem Integrations)
Extensions transform Copilot from a code-completer into a full DevOps/SRE assistant.

### 1. Docker for GitHub Copilot
*   **Use Case:** Containerization & Security.
*   **Commands/Prompts:** `@docker generate a multi-stage Dockerfile for this Spring Boot Maven project.` | `@docker analyze this image for vulnerabilities using Docker Scout.`
*   **Why it's top:** Eliminates the need to memorize Dockerfile syntax and CLI commands.

### 2. Sentry
*   **Use Case:** Production Bug Resolution.
*   **Commands/Prompts:** `@sentry What is causing issue PROJ-123?`
*   **Why it's top:** Directly links production stack traces to your local codebase and suggests the exact lines of code to fix the exception.

### 3. PerplexityAI
*   **Use Case:** Up-to-date Knowledge Retrieval.
*   **Commands/Prompts:** `@perplexity What are the new features in Java 21's Virtual Threads?`
*   **Why it's top:** Bypasses Copilot's knowledge cutoff date by searching the live internet for API docs and solutions.

### 4. Postman
*   **Use Case:** API Development & Testing.
*   **Why it's top:** Send API requests, inspect responses, and generate integration tests directly from the Copilot Chat without opening the Postman desktop app.

### 5. Datadog
*   **Use Case:** Observability & APM.
*   **Commands/Prompts:** `@datadog show me the latency graph for the /api/v1/users endpoint over the last 24 hours.`

### 6. MongoDB / DataStax
*   **Use Case:** Database Query Generation.
*   **Commands/Prompts:** `@mongodb write an aggregation pipeline to group users by country and calculate average spend.`

### 7. Mermaid Chart
*   **Use Case:** Architecture Visualization.
*   **Commands/Prompts:** `@mermaid generate a sequence diagram for the OAuth2 login flow based on my AuthController.java.`

### 8. GitHub Advanced Security (GHAS)
*   **Use Case:** DevSecOps.
*   **Why it's top:** Automatically scans PRs for secrets, SQL injection, and XSS, suggesting inline auto-fixes before merging.

### 9. Azure / AWS Toolkit
*   **Use Case:** Cloud Infrastructure.
*   **Commands/Prompts:** `@aws generate a CloudFormation template to deploy this Lambda function behind an API Gateway.`

### 10. GitBook / Confluence (Docs)
*   **Use Case:** Internal Team Knowledge.
*   **Commands/Prompts:** `@gitbook what is our company's standard pagination format for REST APIs?`

---

## 💻 Part 2: Top 10 Native Copilot Skills (Built-in Power Moves)
These require no installation. They use Copilot's core NLP to AST (Abstract Syntax Tree) engine.

### 11. `@workspace` (Global Context)
*   **Deep Skill:** Don't just ask general questions. Use it for refactoring.
*   **Prompt:** `@workspace I want to migrate from log4j to SLF4J. Find all files using log4j and suggest the necessary refactoring steps.`

### 12. `@terminal` (CLI Command Generation)
*   **Deep Skill:** Complex Bash/PowerShell chaining.
*   **Prompt:** `@terminal find all .log files modified in the last 7 days and compress them into an archive named logs_backup.tar.gz.`

### 13. `/tests` (Advanced Parameterized Testing)
*   **Deep Skill:** Edge-case boundary testing.
*   **Prompt:** `/tests Generate JUnit 5 parameterized tests for this method. Include edge cases for null inputs, empty strings, and negative numbers.`

### 14. `/explain` (Legacy Code Decoder)
*   **Deep Skill:** Reverse-engineering undocumented spaghetti code or complex Regex.
*   **Prompt:** Highlight code -> `/explain what business logic is this SQL query trying to achieve?`

### 15. `/fix` (Inline Auto-Remediation)
*   **Deep Skill:** Fixing IDE linting errors.
*   **Workflow:** When SonarLint or your compiler throws a red underline, highlight it -> `/fix resolve this potential NullPointerException.`

### 16. Multi-Language Translation
*   **Deep Skill:** Porting services.
*   **Prompt:** Highlight a Python script -> `Translate this data processing logic into idiomatic Java using the Stream API.`

### 17. TDD (Test-Driven Development) Reverse Engineering
*   **Deep Skill:** Write the test first, let Copilot write the code.
*   **Workflow:** Write `public void testCalculateTax()` -> Let Copilot fill the test -> Create empty `calculateTax()` method -> Copilot auto-completes the logic to pass the test.

### 18. Natural Language to Regex/Cron
*   **Deep Skill:** Eliminating syntax lookup.
*   **Prompt:** `// Regex to match a valid IPv4 address` or `// Cron expression for every weekday at 3:30 AM`.

### 19. Type Definitions / Schema to Code
*   **Deep Skill:** Generating boilerplate from DB schemas.
*   **Workflow:** Paste a raw SQL `CREATE TABLE` statement -> Prompt: `Generate a Java JPA Entity class for this table with Lombok annotations.`

### 20. Copilot Edits / Agent Mode (Preview)
*   **Deep Skill:** Autonomous multi-file modifications.
*   **Prompt:** `Update the database schema to include a 'status' column, update the JPA entity, add it to the DTOs, and modify the frontend React table to display it.` Copilot edits all files simultaneously.
